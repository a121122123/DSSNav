# DSSNav
Code for "基於動態社交空間之醫療機器人導航系統開發" 

- 安裝docker
- 安裝依賴包
- 修改內容
- 執行模擬步驟

## 安裝docker 

- Use file "dockerfile_fin"
```bash
docker build --build-arg="USER_NAME=$(whoami)" --build-arg="USERID=`id -u`" --build-arg="USERGID=`id -g`" -f dockerfile_fin -t ros_docker .
```

```bash
docker run --privileged --gpus all -ld --name ros_docker --user $(whoami) -v /media:/media -v /home/$(whoami)/host_dir:/home/$(whoami)/docker_dir -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw --env="XAUTHORITY=/tmp/.docker.xauth" --device /dev/snd --env ALSA_CARD=Generic --env="DISPLAY" -h ros_docker -it ros_docker 
```

## 安裝依賴包

- ros安裝 : https://wiki.ros.org/noetic/Installation/Ubuntu 
	- pedsim : https://github.com/srl-freiburg/pedsim_ros
		- pedsim 踩坑：
			QObject’ is an inaccessible base of ‘rviz::_AdditionalTopicSubscriber’
				解決參考：https://github.com/ai-winter/ros_motion_planning/issues/72
	- turtlebot3 simulation : https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/
		- git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
		- cd ~/catkin_ws && catkin_make
		- source devel/setup.bash
			- export TURTLEBOT3_MODEL=burger (option:burger/waffle/waffle_pi)
	- ultralytics_ros : https://github.com/Alpaca-zip/ultralytics_ros?tab=readme-ov-file
		- $ cd ~/{ROS_WORKSPACE}/src
		- $ GIT_LFS_SKIP_SMUDGE=1 git clone -b noetic-devel https://github.com/Alpaca-zip/ultralytics_ros.git
		- $ rosdep install -r -y -i --from-paths .
		- $ python3 -m pip install -r ultralytics_ros/requirements.txt
		- $ cd ~/{ROS_WORKSPACE} && catkin build
	- aws-robomaker-hospital-world : https://github.com/aws-robotics/aws-robomaker-hospital-world/tree/ros1
		- $ sudo bash setup.sh
		- modify launch file 'gui' to true(or add 'gui:=true' when using roslaunch)
	- cartographer : https://hackmd.io/@darren1346/Bk4FMx2oc
		- sudo apt-get update
		- sudo apt-get install -y python-wstool python-rosdep ninja-build stow
		- mkdir catkin_ws
		- cd catkin_ws
		- wstool init src
		- wstool merge -t src https://raw.githubusercontent.com/cartographer-project/cartographer_ros/master/cartographer_ros.rosinstall
		- wstool update -t src
		- sudo rosdep init
		- rosdep update
		- rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y
			- 跳錯到catkin_ws/src/cartographer的資料夾裡的package.xml刪除掉第46行也就是<depend>libabsl-dev</depend>然後再重新執行該指令。
		- src/cartographer/scripts/install_abseil.sh
		- sudo apt-get remove ros-${ROS_DISTRO}-abseil-cpp
		- catkin_make_isolated --install --use-ninja
			- 一直跳錯說找不到abs那包，可能是跑完他腳本後，要跟terminal上的指令次一次
		- source devel_isolated/setup.bash
	- gazebo_ros_2Dmap_plugin : https://github.com/marinaKollmitz/gazebo_ros_2Dmap_plugin/tree/master
		- git clone https://github.com/marinaKollmitz/gazebo_ros_2Dmap_plugin.git
		- 有bug, 不知道為啥xy要一樣大(就是正方形)
	- (x)pgm_map_creator : https://github.com/JZX-MY/pgm_map_creator  (說明影片 : https://www.bilibili.com/video/BV1gz421i7sD/)
		- Not using this package(因為x,y範圍很難調整(結果會很怪)，還有高度也很難調整(跟turtlebot一樣高就是基本上沒東西))
		- git clone https://github.com/JZX-MY/pgm_map_creator
		- follow readme.md
	- people && navigation layer : https://github.com/wg-perception/people/tree/noetic && https://github.com/DLu/navigation_layers/tree/noetic
		- git clone https://github.com/wg-perception/people.git -b noetic
		- git clone https://github.com/DLu/navigation_layers.git -b noetic 
		- rosdep install --from-paths src --ignore-src -y
		- catkin_make
	- teb_local_planner
		- 
- yolo安裝 : pip install ultralytics

## Quickly Guide

- Hospital Scene
	- open gazebo
		- roslaunch launch/gazebo_sim_turtlebot3.launch
	- start tracker
		- roslaunch ultralytics_ros my_tracker_with_cloud.launch yolo_model:=yolov11n_combine.pt
	- start social rule selector
		- roslaunch social_rules_selector social_rule.launch
	- (Optional)start pedsim
		- roslaunch pedsim_simulator my_pedestrians.launch

- Square Scene
	- open gazebo
		- roslaunch pedsim_gazebo_plugin social_contexts.launch
	- spawn robot
		- roslaunch launch/spawn_turtlebot.launch 
	- start tracker 
		- roslaunch ultralytics_ros my_tracker_with_cloud.launch yolo_model:=yolov11n_combine.pt debug:=false map_file:=/home/andre/ros_docker_ws/catkin_ws/map/test.yaml
	- start social rule selector
		- roslaunch social_rules_selector social_rule.launch
	- pause in gazebo
	- start pedsim
		- roslaunch pedsim_simulator simple_pedestrians.launch with_robot:=false
	- start gazebo
	
## 使用說明

- launch the hospital simulation world
	- roslaunch launch/my_launch.launch
	- if need the simulation map
		- rosservice call /gazebo_2Dmap_plugin/generate_map
		- rosrun map_server map_saver -f /home/andre/ros_docker_ws/catkin_ws/map/map /map:=/map2d
- yolo detect
	- yolo訓練 : 
		- 資料準備 : 
			- (目前使用)Deep 3D Perception of People and Their Mobility Aids(http://mobility-aids.informatik.uni-freiburg.de/)
			- (暫未使用，標籤是xml格式)Object detection in hospital facilities: A comprehensive dataset and performance evaluation(https://github.com/Wangmmstar/Hospital_Scene_Data)
		- 資料前處理 : 
			- 不是yolo標籤格式的先轉為yolo標籤格式(transform_annotation.py)
			- 將資料集分為訓練跟驗證(shuffle.py)
		- 準備data.yaml : 
			- 內容為標籤位置和類別
		- 訓練 : 
			- param : 
				- 再補充
	- ultralytics_ros
		- roslaunch ultralytics_ros tracker.launch debug:=true input_topic:=/camera/rgb/image_raw yolo_model:=yolov11n.pt
		- roslaunch ultralytics_ros tracker_with_cloud.launch debug:=true input_topic:=/camera/rgb/image_raw yolo_model:=yolov11n.pt lidar_topic:=/converted_pc camera_info_topic:=/camera/rgb/camera_info
		- roslaunch ultralytics_ros my_tracker_with_cloud.launch
			- Modify default msg (Detection2D/Detection3D) to customized msg (BBox2D/Det3D/Trk3D)
			- Modify launch file for correct param (yolo model/input topic/lidar topic/camera info)
- pedsim
	- generate moving people
		- ref : https://github.com/srl-freiburg/pedsim_ros/blob/master/pedsim_gazebo_plugin/README.md
		-   add this line to your launch file which will open gazebo world, then launch it 
			```   
			<node pkg="pedsim_gazebo_plugin" type="spawn_pedsim_agents.py" name="spawn_pedsim_agents"  output="screen"/>
			```
		- make sure `scenario_file` should be stored in `pedsim_simulator/scenarios/` directory.
			- Scenario_file is a xml file that descript the scene and agents' moves. Here, we ignore the description of scene and just write the agents' moves.
			- launch the pedsim simulate
				```
				roslaunch pedsim_simulator my_pedestrians.launch
				```
		- fix a bug that cause the model spawn in gazebo will be float on the gorund.
			- Find the file `actor_poses_plugin.cpp` in the directory `pedsim_gazebo_plugin`. Modify the parameter `MODEL_OFFSET` from 0.75 to 0.0

- navigation
	- roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=/home/andre/ros_docker_ws/catkin_ws/map/map.yaml
	- if need to control manual
		- roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch