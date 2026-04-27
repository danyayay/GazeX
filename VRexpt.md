# Virtual Reality (VR) Experiment

### 🧪 Experiment Design

The project studies the interaction between a single pedestrian and automated shuttles in shared space using virtual reality. The experiment design includes four variables: 

* `Shuttle behavior`: yielding, non-yielding
* `eHMI presence`: with eHMI, without eHMI
* `Approach angle`: 45, 90, 135 degrees
* `The number of shuttles`: single shuttle, two shuttles with a gap of 3 seconds, two shuttles with a gap of 5 seconds

An illustration of the setup is shown below: 

<p align="center">
  <img width="100%" src="figures/expt_setup.png">
</p>

### 📋 Experiment Procedure

The formal experiment consists of three phases:

1. **Familiarization Phase (Task 1-2)**: Participants practice moving freely in the VR environment and become familiar with shuttle behaviors
2. **Formal Experiment (Task 3)**: Participants navigate through randomized scenarios combining the experimental variables
3. **Post-Experiment Questionnaire**: Collects subjective feedback on safety, trust, and acceptance. Questionnaire is located at `data/questionnaire.pdf`

<p align="center">
  <img width="100%" src="figures/procedure.png">
</p>


### 📊 Data Collection

The system records:
- **HMD Data**: Head location, speed, orientation
- **Eye Gaze Data**: Gaze origin, gaze direction, fixation labels
- **Environment Data**: Vehicle location, speed, eHMI states
- **Behavioral Data**: Participant responses and questionnaire answers




### 🔧 Data Preprocessing
* `utils.preprocessing.py` preprocesses the VR data file for each of the participant (including to extract meaningful variables). 
* `notebooks/processing/process_questionnaire.ipynb` processes all post questionnaire information and saves it as raw. 
* `utils.integrating.py` integrates all VR data files and data tables files into each of their single file, and then do additional preprocessing (i.e., eye gazing interpolation). 
* `notebooks/processing/descriptive_analysis.ipynb` conducts all the descriptive analysis and boxplots. 




## 🎮 Unreal Engine 5 Setup

Before running the project, install the plugin `FileSDK` to your ue5 account and enable all the missing plugins in ue5. 

The project contains three main maps: 
* The persistent level `/Game/Downtown_West/Maps/Demo_Environment` with always-on daylight sublevel `Game/Downtown_West/Maps/Sub-Levels/Daytime_Lighting`.
* A sub-level of the **familiarization** experiment with automated shuttles at `/Game/SharedSpace/Maps/Sub-Levels/FamTaskPodMap`.
* A sub-level of the **formal** experiment with automated shuttles at `/Game/SharedSpace/Maps/Sub-Levels/PodMap`.


Main implementations are in `/Game/SharedSpace/`. You can setup the global variables at `/Game/SharedSpace/MyGameInstance` including: 

*For formal experiment (`task3`):* 
* `ExptCenter`: Coordinate of the intersection point between the first automated shuttle and the pedestrian. 
* `ExptStartSideDistance`: Initial distance between the intersection point and pedestrian's start position (green circle).
* `ExptEndSideDistance`: Initial distance between the intersection point and pedestrian's end position (white circle).
* `FirstPodDistance`: Initial distance between the intersection point and the first automated shuttle (front instead of center). 
* `ExptSetupDT`: Data table for formal experiment, consisting of combinations of the four experimental variables above. 
* `ExptSetupIndex`: Start the formal experiment from the specified index (indexed from 0). 
* `TiltAngle`: Due to space limitation, we tilt participants' view to avoid collision with walls.  
* `LogDataFlag`: Whether log data when running the project. 

*For familiarization phases (`task2`):*
* `FamSetupDT`: Data table for familiarization experiment. 
* `FamSetupIndex`: Start the familiarization experiment from the specified index (indexed from 0). 

Other variables should not be changed without modifying models in ue5. 
