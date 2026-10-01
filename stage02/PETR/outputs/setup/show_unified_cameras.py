"""将本次验证帧的六张真实图像及标定视锥显示在同一参考自车坐标系。"""
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion


def main():
    root = Path(__file__).resolve().parents[2]
    run = root / 'outputs/run20'
    info = sorted(pickle.loads((root / 'v1.0-mini/nuscenes_mini_infos_val.pkl').read_bytes())['infos'],
                  key=lambda x: x['timestamp'])[0]
    names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    colors = ['#2563eb', '#dc2626', '#d97706', '#7c3aed', '#059669', '#0891b2']
    r_le = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    t_le = np.asarray(info['lidar2ego_translation'])
    r_eg = Quaternion(info['ego2global_rotation']).rotation_matrix
    t_eg = np.asarray(info['ego2global_translation'])
    depth = 8.0
    fig = plt.figure(figsize=(17, 8.3))
    ax = fig.add_subplot(121, projection='3d')
    bev = fig.add_subplot(122)
    records = []
    for name, color in zip(names, colors):
        cam = info['cams'][name]
        intrinsic = np.asarray(cam['cam_intrinsic'])
        r_cl = np.asarray(cam['sensor2lidar_rotation'])
        t_cl = np.asarray(cam['sensor2lidar_translation'])
        r_ce = r_le @ r_cl
        t_ce = r_le @ t_cl + t_le
        assert np.allclose(r_ce.T @ r_ce, np.eye(3), atol=1e-6)
        # 独立核对 camera -> camera时刻ego -> global -> 参考时刻ego 的链路。
        r_c_ec = Quaternion(cam['sensor2ego_rotation']).rotation_matrix
        t_c_ec = np.asarray(cam['sensor2ego_translation'])
        r_ec_g = Quaternion(cam['ego2global_rotation']).rotation_matrix
        t_ec_g = np.asarray(cam['ego2global_translation'])
        independent_r = r_eg.T @ r_ec_g @ r_c_ec
        independent_t = r_eg.T @ (r_ec_g @ t_c_ec + t_ec_g - t_eg)
        assert np.allclose(r_ce, independent_r, atol=1e-6)
        assert np.allclose(t_ce, independent_t, atol=1e-6)
        with Image.open(cam['data_path']) as image:
            width, height = image.size
            rgb = np.asarray(image.convert('RGB').resize((64, 36))) / 255.0
        u, v = np.meshgrid(np.linspace(0, width-1, 64), np.linspace(0, height-1, 36))
        pixels = np.stack([u, v, np.ones_like(u)], -1)
        p_camera = pixels @ np.linalg.inv(intrinsic).T * depth
        p_ego = p_camera @ r_ce.T + t_ce
        # 回投影核对图像平面的像素位置，不能将该虚拟深度解释为真实目标深度。
        recovered = (p_ego - t_ce) @ r_ce @ intrinsic.T
        recovered = recovered[..., :2] / recovered[..., 2:3]
        assert np.allclose(recovered, pixels[..., :2], atol=1e-6)
        ax.plot_surface(p_ego[..., 0], p_ego[..., 1], p_ego[..., 2],
                        facecolors=rgb, rstride=1, cstride=1, shade=False,
                        linewidth=0, antialiased=False, alpha=.97)
        corners = p_ego[[0, 0, -1, -1], [0, -1, -1, 0]]
        for corner in corners:
            ax.plot(*np.stack([t_ce, corner]).T, color=color, alpha=.65, linewidth=.7)
        outline = np.vstack([corners, corners[0]])
        ax.plot(*outline.T, color=color, linewidth=1.4)
        center = r_ce @ np.array([0., 0., depth]) + t_ce
        short = name.replace('CAM_', '')
        ax.text(center[0], center[1], center[2]+3, short, color=color, fontsize=8,
                horizontalalignment='center')
        ax.scatter(*t_ce, color=color, s=12)
        # 左右中线构成的视锥截面在xy平面上的投影。
        edge_pixels = np.array([[0., height/2, 1.], [width-1., height/2, 1.]])
        edge_points = (edge_pixels @ np.linalg.inv(intrinsic).T * depth) @ r_ce.T + t_ce
        triangle = np.vstack([t_ce, edge_points, t_ce])
        bev.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=.08)
        bev.plot(triangle[:, 0], triangle[:, 1], color=color, linewidth=1.3)
        bev.scatter(t_ce[0], t_ce[1], color=color, s=25)
        bev.annotate('', xy=center[:2], xytext=t_ce[:2],
                     arrowprops=dict(arrowstyle='->', color=color, linewidth=1.7))
        label = t_ce + r_ce @ np.array([0., 0., 12.5])
        bev.text(label[0], label[1], short, color=color, fontsize=9,
                 ha='center', va='center')
        records.append({'camera': name, 'image_path': str(cam['data_path']),
                        'timestamp_us': int(cam['timestamp']), 'intrinsic': intrinsic.tolist(),
                        'camera_to_reference_ego_rotation': r_ce.tolist(),
                        'camera_origin_reference_ego_m': t_ce.tolist(),
                        'optical_axis_reference_ego': r_ce[:, 2].tolist(),
                        'independent_transform_check': True, 'pixel_round_trip_check': True})
    ax.scatter(0, 0, 0, color='#111827', s=35)
    ax.quiver(0, 0, 0, 3, 0, 0, color='red')
    ax.quiver(0, 0, 0, 0, 3, 0, color='green')
    ax.quiver(0, 0, 0, 0, 0, 3, color='blue')
    ax.set(xlabel='x 向前 / m', ylabel='y 向左 / m', zlabel='z 向上 / m',
           xlim=(-15, 15), ylim=(-15, 15), zlim=(-5, 10), title='六路真实图像映射到统一坐标系中的虚拟图像平面')
    ax.view_init(elev=36, azim=-48)
    ax.set_box_aspect((2, 2, 1))
    bev.scatter(0, 0, color='#111827', s=50, label='参考自车原点')
    bev.annotate('', xy=(4, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#374151'))
    bev.annotate('', xy=(0, 4), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='#374151'))
    bev.set(xlabel='x 向前 / m', ylabel='y 向左 / m', xlim=(-18, 18), ylim=(-18, 18),
            title='六相机位置、光轴与视锥截面的俯视投影')
    bev.set_aspect('equal')
    bev.grid(alpha=.2)
    bev.legend(loc='lower right')
    fig.suptitle('scene-0103 / 验证帧 0：六路相机标定及图像的统一坐标显示', fontsize=16, y=.97)
    fig.text(.5, .04, '共同参考：本帧 LIDAR_TOP 时间戳对应的自车坐标系；x前、y左、z上，单位米。\n'
             '图像平面设定为相机光轴深度8米，仅展示标定关系及像素射线；不代表真实场景深度或点云重建。',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=.035, right=.98, bottom=.14, top=.9, wspace=.15)
    fig.savefig(str(run / 'six_camera_unified_coordinates.png'), dpi=180)
    plt.close(fig)
    record = {'sample_token': info['token'], 'reference_timestamp_us': int(info['timestamp']),
              'reference_frame': 'ego_at_reference_lidar_timestamp', 'axes': 'x forward, y left, z up',
              'unit': 'meter', 'virtual_optical_depth_m': depth, 'actual_depth_reconstruction': False,
              'camera_count': len(records), 'cameras': records}
    (run / 'six_camera_calibration.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Generated unified camera figure and calibration; six transforms and pixel round trips verified.')


if __name__ == '__main__':
    main()
