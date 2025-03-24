# Using NuMicro-NUC140, Combine a Neural Network to Recognize 7 Colors

- In this project, we employ three photoresistors to measure light intensity corresponding to RGB values. Each photoresistor is covered with a translucent filter (red, green, or blue), allowing only the respective color's light to pass through. This setup enables the detection of the intensity of each individual RGB component, which is then analyzed to recognize seven distinct colors based on their light intensity.
- The methodology begins with data collection, followed by training a simple neural network using the RGB values generated by the photoresistors. The trained model produces a weight matrix, which is then utilized in a real-time predictive model for accurate color recognition.

<img src="https://github.com/user-attachments/assets/5a11385c-0df4-416a-b158-6b6426f52cee" width="650">

## Content
* [Device and IDE](#device-and-ide)
* [Circuit Diagram](#circuit-diagram)
* [Operating Principle](#operating-principle)
* [Machine Learning](#machine-learning)
* [STEPS](#steps)
* [DEMO Video (YouTube)](#demo-video-youtube)

## Device and IDE
### [NuvoTon NuMicro-NUC140](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m0-mcus/nuc140-240-connectivity-series/?__locale=zh_TW)
<img src="https://github.com/user-attachments/assets/30c15078-53bf-48de-983a-7b072c37de7f" width="500" />

### [keil uVision 5](https://www.keil.com/download/list/uvision.htm)
<img src="https://github.com/user-attachments/assets/f8142b6a-61fc-459c-83a1-2361ee4d3eb9" width="200">

## Circuit Diagram
![image](https://github.com/user-attachments/assets/26526984-e303-4d4f-978f-6cae3745e3bc)

## Method
### STEPS
1. `Sensor`:
The photoresistors change their resistance based on the intensity of light, which affects the voltage across them, which will be read by the ADC (Analog-to-digital converter) channels on the MCU.
2. `Analog-to-Digital Conversion (ADC)`:
The MCU reads the analog voltage values from the three photoresistors via the ADC channels.
    - Each channel reads one of the RGB values (e.g., R for Red, G for Green, and B for Blue).
    - The ADC converts these analog voltage signals into corresponding digital values for further processing.
3. `Neural Network Prediction`:
The RGB values are input into the trained neural network model.
    - The neural network processes the values and classifies them into one of seven categories (Background, Blue, Magenta, Red, Orange, Yellow, Green).
    - The network outputs the predicted color based on the provided RGB inputs.
4. `Communication with UART`:
The predicted classification result (the identified color) is sent through UART communication.
    - The PL2303HXA USB-to-serial adapter is used to transmit the data from the MCU to an external device (e.g., a PC or terminal).
5. `Output Display`:
The classification result is displayed on a terminal connected via UART.
    - The user can observe the predicted color result in real-time.
6. `Repeat the Process`:
The system continuously reads new RGB values from the photoresistors, processes them through the neural network, and displays updated predictions.
    - This cycle allows for real-time color classification.
      
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
- ![image](https://github.com/user-attachments/assets/e0ff4258-7df9-4501-815c-37fd3775d799)


### Machine Learning
- Use MLP (Multilayer Perceptron) as the model architecture.
![image](https://github.com/user-attachments/assets/987c1b44-1aa4-425b-8db1-970c32820f4c)






## DEMO Video (YouTube) 
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/bp_OSHvo8vk/0.jpg)](https://www.youtube.com/watch?v=bp_OSHvo8vk)

