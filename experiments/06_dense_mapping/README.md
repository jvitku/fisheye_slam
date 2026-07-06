# Track E — Sensor-pod output stage: position + dense 3D map

The sensor pod's deliverable is **odometry (position) + a local dense 3D map**.
The pipeline that produces it from pod data (bag or live):

```
pod bag (cams + /uav1/sensor_pod/imu)
   └─> VIO candidate (Track A/B/C winner)      -> pose stream   [output 1]
        └─> multi-view fisheye depth            -> depth images / point clouds
             (adjacent-pair stereo; the triangle pod adds vertical baselines)
             └─> TSDF fusion                    -> mesh + ESDF  [output 2]
                  voxblox (CPU / Pi 5)  or  nvblox (GPU / Orin)
```

## Status
- [ ] voxblox image builds (`docker/voxblox/`)
- [ ] Pipe a TUM-VI or sim run through: candidate poses + naive stereo depth -> TSDF mesh
- [ ] Depth from fisheye pairs (rectify-to-pinhole first pass; sphere-sweeping later)
- [ ] Map-quality metric vs sim ground truth (Isaac scene mesh) — accuracy/completeness
- [ ] nvblox variant on Orin (Phase 3)

## Notes
- **Depth is the hard part**, not fusion. First pass: undistort adjacent-pair
  fisheye to virtual pinhole (tools/fisheye models), OpenCV SGBM, feed
  voxblox `pointcloud + pose`. That's deliberately crude — it establishes the
  interface; quality work comes after a VIO core is chosen.
- The 3-cam triangle pod exists exactly for this experiment: horizontal AND
  diagonal baselines -> depth on horizontal edges too (where a horizontal-only
  stereo pair fails).
- voxblox wants: `sensor_msgs/PointCloud2` + `geometry_msgs/TransformStamped`
  (or TF). Feed poses from the candidate output, NOT ground truth, unless
  isolating mapping error deliberately.
- Map benchmark idea (Phase 2+): export the Isaac scene mesh, sample GT point
  cloud, compare TSDF mesh (accuracy = est->GT distances, completeness =
  GT->est). Add to bench/evaluate.py as a second evaluator when depth works.
