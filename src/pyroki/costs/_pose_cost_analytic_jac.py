import jax
import jax.numpy as jnp
import jaxlie
import jaxls

from .._robot import Robot


def _get_joints_applied_to_target(
    robot: Robot,
    target_joint_idx: jax.Array,
):
    """Get the indices of the actuated joints that affect the pose of the target link."""

    # Get kinematic chain indices.
    def body_fun(i, curr_parent, idx):
        act_joint_idx = jnp.where(
            i == curr_parent,
            robot.joints.actuated_indices[i],
            -1,
        )
        curr_parent = jnp.where(
            i == curr_parent,
            robot.joints.parent_indices[i],
            curr_parent,
        )
        return (
            i - 1,
            curr_parent,
            idx.at[i].set(
                jnp.where(
                    i > target_joint_idx,
                    -1,
                    act_joint_idx,
                )
            ),
        )

    idx_applied_to_target = jnp.zeros(robot.joints.num_joints, dtype=jnp.int32)
    idx_applied_to_target = jax.lax.while_loop(
        lambda carry: (carry[0] >= 0),
        lambda carry: body_fun(*carry),
        (robot.joints.num_joints - 1, target_joint_idx, idx_applied_to_target),
    )[-1]
    return idx_applied_to_target


_PoseCostJacCache = tuple[jax.Array, jax.Array, jaxlie.SE3]


def pose_cost_analytic_jac(
    robot: Robot,
    joint_var: jaxls.Var[jax.Array],
    target_pose: jaxlie.SE3,
    target_link_index: jax.Array,
    pos_weight: jax.Array | float,
    ori_weight: jax.Array | float,
) -> jaxls.Cost:
    assert target_link_index.shape == (), (
        "Analytical jacobian currently only supports single target link."
    )

    base_link_mask = robot.links.parent_joint_indices == -1
    parent_joint_indices = jnp.where(
        base_link_mask, 0, robot.links.parent_joint_indices
    )
    target_joint_idx = parent_joint_indices[target_link_index]
    joints_applied_to_target = _get_joints_applied_to_target(robot, target_joint_idx)

    def pose_cost_jac(
        vals: jaxls.VarValues,
        jac_cache: _PoseCostJacCache,
        robot: Robot,
        joint_var: jaxls.Var[jax.Array],
        target_pose: jaxlie.SE3,
        target_link_index: jax.Array,
        pos_weight: jax.Array | float,
        ori_weight: jax.Array | float,
    ) -> jax.Array:
        """Jacobian for pose cost with analytic computation."""
        del vals, joint_var, target_pose  # Unused!
        Ts_world_joint, Ts_world_link, pose_error = jac_cache

        T_world_ee = jaxlie.SE3(Ts_world_link[target_link_index])
        Ts_world_joint = jaxlie.SE3(Ts_world_joint)

        R_ee_world = T_world_ee.rotation().inverse()

        # Get joint twists
        joint_twists = robot.joints.twists

        # Get angular velocity components (omega)
        omega_local = joint_twists[..., 3:]
        omega_wrt_world = Ts_world_joint.rotation() @ omega_local
        omega_wrt_ee = R_ee_world @ omega_wrt_world

        # Get linear velocity components (v)
        vel_local = joint_twists[..., :3]
        vel_wrt_world = Ts_world_joint.rotation() @ vel_local

        # Compute the linear velocity component (v = ω × r + v_joint)
        vel_wrt_world = (
            jnp.cross(
                omega_wrt_world,
                T_world_ee.translation() - Ts_world_joint.translation(),
            )
            + vel_wrt_world
        )
        vel_wrt_ee = R_ee_world @ vel_wrt_world

        # Combine into spatial jacobian
        jac = jnp.where(
            joints_applied_to_target[:, None] != -1,
            jnp.concatenate(
                [
                    vel_wrt_ee,
                    omega_wrt_ee,
                ],
                axis=1,
            ),
            0.0,
        ).T
        jac = pose_error.jlog() @ jac

        # TODO: @cmk I don't really know how to correctly use these 🥲
        #
        # this indexing/slicing here works for Panda but I'm not sure how to
        # make this generalize
        jac = jac[:, robot.joints.actuated_indices][
            :, : robot.joints.num_actuated_joints
        ]

        # Apply weights
        weights = jnp.array([pos_weight] * 3 + [ori_weight] * 3)
        return jac * weights[:, None]

    @jaxls.Cost.create_factory(jac_custom_with_cache_fn=pose_cost_jac)
    def pose_cost(
        vals: jaxls.VarValues,
        robot: Robot,
        joint_var: jaxls.Var[jax.Array],
        target_pose: jaxlie.SE3,
        target_link_index: jax.Array,
        pos_weight: jax.Array | float,
        ori_weight: jax.Array | float,
    ) -> tuple[jax.Array, _PoseCostJacCache]:
        """Computes the residual for matching link poses to target poses."""
        assert target_link_index.dtype == jnp.int32
        joint_cfg = vals[joint_var]

        Ts_world_joint = robot._forward_kinematics_joints(joint_cfg)
        Ts_world_link = robot._link_poses_from_joint_poses(Ts_world_joint)

        T_world_ee = jaxlie.SE3(Ts_world_link[..., target_link_index, :])
        pose_error = target_pose.inverse() @ T_world_ee
        return (
            pose_error.log() * jnp.array([pos_weight] * 3 + [ori_weight] * 3),
            # Second argument is cache parameter, which is passed to the custom Jacobian function.
            (Ts_world_joint, Ts_world_link, pose_error),
        )

    return pose_cost(
        robot, joint_var, target_pose, target_link_index, pos_weight, ori_weight
    )
