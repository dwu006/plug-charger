import os

import mujoco
import mujoco.viewer
import numpy as np

from teledex import Session, MujocoHandler

WRIST_SCALE = 1.0


def main() -> None:
    here = os.path.dirname(__file__)
    model = mujoco.MjModel.from_xml_path(os.path.join(here, "SO101", "scene.xml"))
    data = mujoco.MjData(model)

    def site_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    def joint_id(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    target_site_id = site_id("gripperframe")
    teleop_site_id = site_id("teleop_target")

    ik_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    dof_ids = [int(model.jnt_dofadr[joint_id(n)]) for n in ik_joint_names]
    qpos_ids = [int(model.jnt_qposadr[joint_id(n)]) for n in ik_joint_names]

    g_jid = joint_id("gripper")
    g_qpos_id = int(model.jnt_qposadr[g_jid])
    GRIPPER_OPEN = float(model.jnt_range[g_jid][0])
    GRIPPER_CLOSED = float(model.jnt_range[g_jid][1])

    session = Session(debug=True)

    mujoco.mj_forward(model, data)
    ref_site_pos = data.site_xpos[target_site_id].copy()

    R_post = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    handler = MujocoHandler(model=model, data=data)
    session.add_handler(handler)
    handler.link_site(
        name="teleop_target",
        scale=0.5,
        position_origin=(R_post.T @ ref_site_pos).tolist(),
        post_transform=[
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ],
    )

    # seed the site so IK doesn't chase (0,0,0) before first phone packet
    model.site_pos[teleop_site_id] = np.array(ref_site_pos)

    session.start()

    gripper_closed = False
    last_toggle = None
    rot_ref = None

    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            print("viewer running, connect teledex app to start")
            while viewer.is_running():
                latest = session.get_latest_data()
                pos = latest.get("position")
                if pos is None:
                    pos = latest.get("position_hand")
                rotation = latest.get("rotation")
                if rotation is not None and rot_ref is None:
                    rot_ref = np.array(rotation, dtype=float)

                if pos is not None:
                    target_pos = model.site_pos[teleop_site_id].copy()

                    mujoco.mj_forward(model, data)
                    current_pos = np.array(data.site_xpos[target_site_id], dtype=float)
                    delta = target_pos - current_pos

                    if np.linalg.norm(delta) > 1e-5:
                        jacp = np.zeros((3, model.nv))
                        jacr = np.zeros((3, model.nv))
                        mujoco.mj_jacSite(model, data, jacp, jacr, target_site_id)
                        J = jacp[:, dof_ids]

                        lam = 1e-3
                        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), delta)
                        step = np.linalg.norm(dq)
                        if step > 0.05:
                            dq *= 0.05 / step

                        for qid, dq_i in zip(qpos_ids, dq):
                            data.qpos[qid] += float(dq_i)

                    if rotation is not None and rot_ref is not None:
                        R_rel = rot_ref.T @ np.array(rotation, dtype=float)
                        pitch = np.arctan2(-R_rel[2][0], np.sqrt(R_rel[2][1]**2 + R_rel[2][2]**2))
                        wf_qid = qpos_ids[3]
                        wf_jid = joint_id("wrist_flex")
                        data.qpos[wf_qid] = float(np.clip(
                            data.qpos[wf_qid] + pitch * WRIST_SCALE,
                            model.jnt_range[wf_jid][0],
                            model.jnt_range[wf_jid][1],
                        ))

                toggle = latest.get("toggle")
                if last_toggle is not None and toggle != last_toggle:
                    gripper_closed = not gripper_closed
                last_toggle = toggle
                data.qpos[g_qpos_id] = GRIPPER_CLOSED if gripper_closed else GRIPPER_OPEN

                mujoco.mj_step(model, data)
                viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()


if __name__ == "__main__":
    main()
