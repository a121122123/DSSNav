# DSSNav
Code for "基於動態社交空間之醫療機器人導航系統開發" 


## Build Docker 
Build docker image
```bash
docker build -f dockerfile -t ros-noetic-gpu .
```

(Optional)For using gui in docker 
```bash
xhost +local:root
```

Run docker
```bash
docker run --gpus all -it --name test -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw --env="XAUTHORITY=/tmp/.docker.xauth" --device /dev/snd --env ALSA_CARD=Generic --env="DISPLAY" -v $(pwd):/workspace ros-noetic-gpu
```

## Quickly Demo

- Hospital Scene
	```bash
	cd catkin_ws
	apt update
	rosdep install --from-paths src --ignore-src -r -y
	source env_setup.sh
	./start_simulation_hospital.sh [NUM_RUNS] [DEBUG_MODE] [USE_TEB] [ALWAYS_STATIC]
	```
	- NUM_RUNS: Input numbers. Total number of simulations executed.
	- DEBUG_MODE: Input true/false. To visualize gui or not.
	- USE_TEB: Input true/false. If true, use teb as local planner planner; otherwise, use dwa.
	- ALWAYS_STATIC: Input true/false. If true, use `dynamic social space`; otherwise, use static.