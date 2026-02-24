# Day 1 - 2/18/2026
Finished the 147 midterm today. I didn't really study param update calculations and like optimizer stuff so i was lost TT. Project approved by Prof Kao. Task: Collect data using LeRobot SO-100 arm and Implement CNN + Transformer/LSTM architecture to generate robot actions. Possible tasks include pick & place, plug charger, or pouring. 

- Setup conda env with lerobot
- lerobot-find-port # find port which arm is connected to
sudo chmod 666 /dev/ttyACM*
- lerobot-calibrate --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm 
- lerobot-calibrate --teleop.type=so100_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm 

Calibration having issues. Missing motors seemed to havea missing wire. Leader arm can calibrate after u use less output power cord.

# Day 2 - 2/19/2026

Got wire from lab and wired up the gripper for the follower arm. 
Follower Arm Errors:
Missing motor IDs:
  - 1 (expected model: 777)
  - 6 (expected model: 777)

Full expected motor list (id: model_number):
{1: 777, 2: 777, 3: 777, 4: 777, 5: 777, 6: 777}

Full found motor list (id: model_number):
{2: 777, 3: 777, 4: 777, 5: 777}

lerobot-setup-motors --robot.type=so100_follower --robot.port=/dev/ttyACM0

Connect the controller board to the 'gripper' motor only and press enter.
'gripper' motor id set to 6
Connect the controller board to the 'wrist_roll' motor only and press enter.
'wrist_roll' motor id set to 5
Connect the controller board to the 'wrist_flex' motor only and press enter.
'wrist_flex' motor id set to 4
Connect the controller board to the 'elbow_flex' motor only and press enter.
'elbow_flex' motor id set to 3
Connect the controller board to the 'shoulder_lift' motor only and press enter.
'shoulder_lift' motor id set to 2
Connect the controller board to the 'shoulder_pan' motor only and press enter.
'shoulder_pan' motor id set to 1

But after calibration still got missing motors for some of them. Will probably try rerouting them tmr. 

Downloaded test lerobot data (videos and parquert). Implemented an extract frames which will captures frames in 30 FPS

# Day 3 - 2/20/2026

- Extract frames work
- Parquet file can be kept for training
- Research how the CNN architecture will work
- Add a basic cnn.py class that uses resnet18 backbone but changed last 2 layers

# Day 4 - 2/21/2026

- Switch new follower arm and calibration works
- Leader arm calibration and motor kind of breaking for some reason, can't teleop or record 

lerobot-teleoperate --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm --teleop.type=so100_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm

lerobot-record --robot.type=so100_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm --teleop.type=so100_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm --dataset.repo_id=dwux/test_lerobot --dataset.num_episodes=1 --dataset.single_task="Plug the charger" --dataset.push_to_hub=false

- Test realsense camera but doesn't really work, keeps breaking in between. Figured I need USB 3.0 and this old laptop doesn't have it will have to use my newer laptop. 

# Day 5 - 2/22/2026

lerobot-setup-motors --teleop.type=so100_leader --teleop.port=/dev/ttyACM0

Tried Configuring setting up another leader arm. Doesn't seem to work at all. Even after resetting up the motors. I'm thinking of using teledex (control via phone) made by one of my labmates. 

# Day 6 - 2/23/2026

- Found lerobot has its own phone teleop setup
- Use lerobot's teleop with phone via the HEBI mobile IO app and the teleoperate script lerobot and able to control the arm
- The control folder contains the lerobot phone teleop scripts and the urdf from TheRobotStudio, updated it to set the "name" and "family". 
- Update the record script so we can choose when it starts to record: s to start, y to save, and r to retry the episode. 
- Teleop is really hard to control with phone. Not exactly sure which direction for which for the robot.
- Using a USB 3.0 port with realsense which works and test record camera - after some digging better to collect data in videos rather than frames

# Day 7 - 2/24/2026

