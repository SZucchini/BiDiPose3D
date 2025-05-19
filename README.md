# BiDiPose3D: Binocular Diffusion Pose Estimation for 3D Human Motion Analysis

2視点の2次元関節位置座標系列およびカメラの内部パラメータから3次元の関節位置座標の推定を目指すプロジェクト

## 暫定的なデータ形状

- 2次元関節位置座標: (T, J, C) x 2 view
  - T: フレーム数、81が基本
  - J: 関節数、17が基本
  - C: 座標x, y（+信頼スコアs）、つまりC = 2 or 3
- 2次元関節位置座標: (T, J, 3)
  - T: フレーム数、81が基本
  - J: 関節数、17が基本
  - C: 座標x, y, z
  - 基準カメラから見たカメラ座標系の座標（zが奥行きになる）
- 内部パラメータ行列: (3, 3) x 2
  ```math
  K = 
  \begin{pmatrix}
    f_x & s   & c_x \\
    0   & f_y & c_y \\
    0   & 0   & 1
  \end{pmatrix},
  ```
  
  - f_x, f_y：水平方向・垂直方向の焦点距離（画素単位）  
  - c_x, c_y：主点（画像中心）の画素座標  
  - s：スキュー（通常 0）
- Essential行列: (3, 3)
  ```math
  E = [t]_\times R,\quad
  [t]_\times = \begin{pmatrix}
  0 & -t_z & t_y\\
  t_z & 0 & -t_x\\
  -\,t_y & t_x & 0
  \end{pmatrix},\quad
  \mathbf x_2^\top E\,\mathbf x_1 = 0
  ```
  - E: エッセンシャル行列、2つのカメラの関係性を記述（これがわかればスケールは無視しているが三角測量が可能に）
  - R: カメラ間の相対回転行列
  - t: カメラ間の相対並進ベクトル（方向のみ、スケール不定）  
  - x1とx2は内部パラメータによって正規化されたそれぞれのビューの対応点の同次座標
  - 最後の等式をエピポーラ制約という

## 参考情報

- [Extrinsics Camera Calibration from a moving person](https://vision.ist.i.kyoto-u.ac.jp/pubs/SLee_RAL22.pdf)（再投影誤差などを用いたバンドル最適化）
- 今日紹介した先行研究2つ
  - [Dual-Diffusion for Binocular 3D Human Pose Estimation](https://proceedings.neurips.cc/paper_files/paper/2024/file/8ea50bf458f6070548b11babbe0bf89b-Paper-Conference.pdf)
  - [Multiple View Geometry Transformers for 3D Human Pose Estimation](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_Multiple_View_Geometry_Transformers_for_3D_Human_Pose_Estimation_CVPR_2024_paper.pdf)
