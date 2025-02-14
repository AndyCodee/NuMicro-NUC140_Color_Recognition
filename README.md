# Using NuMicro-NUC140, Combine a Neural Network to Recognize 7 colors

- In this project, we aim to utilize three photoresistors to measure the light intensity corresponding to RGB values. This is achieved by covering each photoresistor with a red, green, or blue translucent filter, allowing only the respective color’s light to pass through. By doing so, we can capture the intensity of each RGB component and analyze variations in light intensity to identify color pattern changes on banknotes.
- The overall methodology involves first collecting data, then using this time-series data to train a simple neural network. The trained model outputs a weight matrix, which is subsequently used in the final predictive model for real-time banknote color pattern recognition.
<img src="https://github.com/user-attachments/assets/3660d42c-c90f-4c90-97cd-460a651b6222" width="600">





## Content
* [Device and IDE](#device-and-ide)
* [Circuit Diagram](#circuit-diagram)
* [Operating Principle](#operating-principle)
* [Machine Learning](#machine-learning)
* [STEPS](#steps)
* [Flow Chart](#flow-chart)
* [DEMO Video (YouTube)](#demo-video-youtube)

## Device and IDE
### [NuvoTon NuMaker-M032KG](https://direct.nuvoton.com/tw/numaker-m032kg)
![numaker-m032kg](https://github.com/user-attachments/assets/53c2646d-d427-4818-993f-16b76a3c903f)

### [keil uVision 5](https://www.keil.com/download/list/uvision.htm)
<img src="https://github.com/user-attachments/assets/f8142b6a-61fc-459c-83a1-2361ee4d3eb9" width="200">


## Circuit Diagram
![image](https://github.com/user-attachments/assets/65bda780-0380-43b8-a790-017e51f4f0d4)

## Method
### Operating Principle
`Color Sensor`:
- When an object is illuminated with white light, it reflects 
the wavelengths of light that correspond to its inherent 
color.
- Reflected light can be used to determine an object’s color.
- A photoresistor’s resistance varies with light intensity
- Cellophane can assist in filtering light
- Combining these helps detect the intensity of specific 
colored light
- ![image](https://github.com/user-attachments/assets/d53c70fb-e76b-4c94-8a99-273e58268411)

### Machine Learning
- Use MLP (Multilayer Perceptron) as the model architecture.
- ![image](https://github.com/user-attachments/assets/1208b72e-beb0-4a88-8600-db722dd545b8)

### STEPS
1. `Baseline Initialization`:
Perform an initial scan of the ADC values from the three RGB photoresistors to establish the baseline ambient light levels.
2. `Continuous Monitoring`:
Continuously monitor the ADC(Analog-to-digital converter register) values of the RGB photoresistors for any significant changes.
3. `Banknote Detection (Start of Recording)`:
When a significant change in the RGB photoresistor values is detected, it indicates that a banknote has been inserted.
Start recording the variations in RGB intensity.
4. `Banknote Exit Detection (End of Recording)`:
When a second significant change is detected, it indicates that the banknote has passed through, and the ambient light has returned to the baseline.
Stop recording and finalize the RGB time-series data.
5. `Data Resampling`:
Resample the recorded time-series data to a fixed temporal length to ensure consistency in model input.
6. `Banknote Recognition`:
Feed the resampled data into the predictive model to determine the denomination of the banknote.
7. `Result Display`:
Output the recognition result on the terminal.

### Flow Chart
<img src="https://github.com/user-attachments/assets/c9f31a80-07d1-4316-83b5-332495b21d1b" width="700">


## DEMO Video (YouTube) 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/qb9uLU0ng0Y/0.jpg)](https://www.youtube.com/watch?v=qb9uLU0ng0Y)

