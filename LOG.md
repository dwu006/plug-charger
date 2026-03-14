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

- Tried using the so100 urdf for the teleop script - didnt work
- Might build my own phone teleop app ...

# Day 8 - 2/25/2026

- Got midterm back, did way better than expected

# Day 9 - 2/26/2026 to Day 15 - 3/4/2026

- Busy with iros

# Day 16 - 3/5/2026

- Added lerobot support to Teledex (https://github.com/omarrayyann/TeleDex) to control lerobot

# Day 17 - 3/6/2026

- Improved cnn.py with layer freezing methods and better dropout
- Implemented transformer with TransformerEncoder with positional encoding

# Day 18 - 3/7/2026

- Fixed an issue with the lerobot teleop where it wasn't moving the wrist flex
- Tried recording an episode and replaying it (worked!)
- Combined transformer and cnn into our policy: cnn --> transformer --> generate actions (5 joints + gripper)

# Day 19 - 3/8/2026

- Wrote load dataset to load all our joint data and other metadata in the parquet into dataloader
- train.py - our training script for our cnn+transformer policy
- camera recording at 640x480 for rgbd

# Day 20 - 3/9/2026

- Phone teleop wasnt precise enough, switched to lerobot keyboard teleop
- Collected 10 ep of plugging charger, trained a model, wrote deploy script and deployed
- Found it sucked, probably problem in code but realized task it too hard with bad HW, switching to just pick and place
- Added lstm policy pipelines 

# Day 21 - 3/10/2026

- Collected 50ish episodes of pick and place task with keyboard teleop
- Realized data was corrupted because i didn't close the file which was still writing and i killed the script
- Recollected data and trained 50 episodes, deployed — robot reaches but freezes after picking up block
- Realized my actions joints were raw positions so fix that recollect
- Trained and redeployed - failed
- Change my setup so the container is on the side not block most of the top down cam
- Trained and deployed - kind of working (like can do parts with help) 
- Switched to z-score normalization (mean/std per joint), saves norm_stats.npz with dataset
- Added action chunking (similar to act) — model predicts K future actions instead of 1 
- Trained and redeployed - had one sucessful run but had to reposiiton obj
- decided just focus on transformer (no lstm)

# Day 22 - 3/11/2026

- create data augment script to add or remove brightness and vary the copied joint data a bit 
- Training now uses AdamW with weight decay, warmup + cosine schedule, early stopping, grad clipping
- Added smoothness loss (0.1 weight) to encourage smooth action chunks
- Two-phase training: phase 1 with frozen CNN features, phase 2 fine-tunes CNN on raw images
- Tested diff chunk size 4 i think works best for this 
- Add chunk blend for smoother transitions

# Day 23 - 3/12/2026

- More training and deployment
- Run model for results
- Get the loss curves 
- Tested data augmentation
- Worked on report

# Day 24 - 3/13/2026

- Final deployment tests 
- Worked on report


# Day 25 - 3/14/2026

- Cleaned up code
- Finished report