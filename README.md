<div align="left">
  <img src="report_figures/swish_logo.png" alt="DSwish Logo" width="20%">
</div>

# Acoustic Classification of Basketball Shots using TinyML

**Author**: Elliot Hills (WXWX9)

**Edge Impulse:** https://studio.edgeimpulse.com/public/979083/latest

**GitHub Repo:** https://github.com/tri4ngl3/CASA0018-Swish-Acoustic-Shot-Tracker/

**Experiment Logs, Code and Stats:** https://github.com/tri4ngl3/CASA0018-Swish-Acoustic-Shot-Tracker/experiment_logs_and_outputs

**Deployment Code:** https://github.com/tri4ngl3/CASA0018-Swish-Acoustic-Shot-Tracker/casa_deployment.ino

## Introduction
Gathering statistics from regular shooting drills is a key way that basketball players measure shot improvement (Cleary & Zimmerman, 2001). However, mentally tracking makes and misses is challenging and creates a cognitive-motor dual-task that impairs performance (Moreira et al., 2021). Existing automated solutions are imperfect: wearable tech (e.g. ShotTracker) can impact shooting mechanics (Li and Zhang, 2022) and computer vision apps (e.g. HomeCourt and Ballogy) require impractical tripod setups and are invasive in public spaces due to human by-catch (Sandbrook et al., 2018).

This project, Swish, overcomes these problems by using machine learning to classify shots based on the sound of ball-basket interactions. By using audio, Swish bypasses the need for invasive filming or wearable hardware, providing a portable device that can be placed discretely under the rim, leaving the athlete to focus on their shot.

## Application Overview
The Swish system employs an edge machine learning architecture to classify shot audio locally in real-time. The hardware is lightweight and pocket-sized, consisting of an Arduino Nano 33 BLE Sense secured to a portable battery (Figure 1). Audio is continuously captured and partitioned using a a 0.5-second inference stride and a 1.5-second sliding window to capture both the rim-impact of a shot and the subsequent bounce, as both are indicative of shot outcome.

<div align="center">

<img src="report_figures/dual_swish_diagram.png" alt="Overview of Swish showing both court placement and hardware close-up." width="100%">

</div>

**Figure 1.** _The Swish shot tracker. **(A)** Deployment context. Positioning of the Swish sensor on the ground behind the court perimeter, beneath the hoop is shown. As well as the primary acoustic target zone (rim and net) and the downward propagation of sound to the microphone **(B)** Close-up of the assembled prototype detailing the components: (a) the **Arduino Nano 33 BLE Sense microcontroller**, (b) the **5V rechargeable battery**, (c) the **MP34DT05 microphone**, (d) the **RGB LED**, (e) the **custom fabric securement sleeve**, and (f) the **USB power cord** connecting the battery supply to the board._

Each window is processed through a Mel-filterbank energy (MFE) digital signal processing block to convert the audio data into a Mel-spectrogram. This is passed to a quantised Convolutional Neural Network (CNN) classifier, which generates confidence scores for three target classes: make, miss, and noise. An algorithm then identifies the class with the highest score that exceeds the precision-recall tuned confidence threshold. If the window is classified as noise, the device continues listening. However, if a make or a miss is detected, then the respective event is recorded and a 2-second refractory period is initiated to prevent duplicate classifications. The onboard RGB LED then changes from blue, which indicates the device is listening, to red for a miss or green for a make. Shooting statistics are stored in local memory. This data can then be retrieved at the end of a session by connecting via a Bluetooth Low Energy (BLE) UART text interface that links the user's smartphone to the device using the Serial Bluetooth Terminal app.

![Block diagram showing the flow of data through the Swish system](report_figures/swish_application_graphic.png)

**Figure 2.** *The Swish system architecture and data processing pipeline.*

## Data
To ensure training data was tailored to the intended device deployment, a custom dataset was recorded, matching device deployment conditions.

Initial data collection using the Arduino and Edge Impulse's (EI) labelling feature was abandoned due to 20-second recording limits and EI's inefficient labelling workflow. Instead, continuous shooting sessions that included a diverse range of shots were recorded on a smartphone. To mimic the Arduino mic's acoustic profile, the RecForge app was used to disable automatic gain control and record through a single mic the and files were downsampled to 16 kHz and augmented with artificial noise on EI.

To streamline annotation, delayed audio-tagging was used, whereby class labels were stated aloud following a pause after each shot. In Audacity, 1.5 second windows were labelled and audio tags removed (Figure 3) to extract 1,509 data points total, including 353 makes, 572 misses and 584 background noise windows. 

![Image showing Audacity labelling system](report_figures/audacity_casa_diag.png)

**Figure 3.** *Audacity workspace demonstrating the manual audio annotation workflow. Over 2.5 hours of continuous raw shooting audio was segmented into discrete class windows using this method. Noise segments were intentionally left longer, allowing the EI pipeline to automatically partition them into multiple 1.5-second windows.*

In order to correct class imbalance that skewed model predictions, misses were down-sampled to match the exact number of makes. Audio windows were processed in EI into exact 1.5 second windows and Mel-spectrograms, ready for model training (Figure 4).

<div align="center">
  <img src="report_figures/shot_MFE_spectrograms_diagram.png" alt="Diagram showing MFE spectrograms of different shots" width="60%">
</div>

**Figure 4.** *Mel-spectrograms illustrating the acoustic profiles of a clean swish, an unclean make, a miss and background noise (ball bouncing) over a 1.5-second sample window.*

## Model & Experiments
Constructing a TinyML model means navigating the inherent trade-off balance between classification accuracy and processing efficiency (Lin et al., 2020).

### Model Architecture

To determine the optimal model for my device, I initially experimented with three architectures: a standard Dense Neural Network (DNN), a 1D Convolutional Neural Network (CNN), and a 2D CNN (Table 1). To prevent models from overfitting, a high dropout rate was used and noise injection and time-masking were employed to artificially expand the variance of the training dataset.

<div align="center">

| Model Architecture | Validation Accuracy | Test Accuracy | Test F1 Score | Inference Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Dense (DNN)** | 60.3% | 53.38% | 0.53 | 10 ms |
| **1D CNN** | 86.4% | 78.70% | 0.91 | 13 ms |
| **2D CNN** | 87.7% | 83.46% | 0.92 | 306 ms |

</div>

**Table 1.** *Performance metrics across neural network architectures. See logs and screenshots in experiment_logs_and_outputs*

The DNN performed poorly, achieving only 53.38% test accuracy, likely because flattening the spectrogram destroyed the spatial information required to distinguish between classes. The CNNs performed significantly better, with the 2D architecture slightly outperforming the 1D model across all evaluation metrics. The 1D CNN exhibited a notably lower test accuracy, which confusion matrix analysis revealed was primarily driven by a higher proportion of 'uncertain' classifications compared to 2D. However, the 1D CNN was significantly more efficient, executing inference in just 13 ms compared to the 2D CNN at 306 ms. 

Navigating this accuracy-latency trade-off required prioritizing the specific functionality of the application. To operate effectively as a shot tracker, Swish must have a high classification accuracy to reliably capture the minute, incremental improvements a player makes to their shooting percentage over time. As a result, the 1D architecture was deemed insufficiently accurate for the use case and it was decided that the 2D CNN should be used so long as further system optimisation could ensure total latency remained below the 500 ms inference stride.

###  Optimisation

In the baseline system using auto-tuned parameters from EI (16KHz, 40 filters, 512 FFT length, 0.01s stride), the MFE processing block had a high latency of 763 ms. Combined with the 306 ms inference time, the total pipeline latency far exceeded the 500 ms inference stride constraint at 1,069 ms.

To resolve this, three experiments were run to evaluate the effect of downgrading parameters (Table 2): 
1. Downsampling the audio sample rate
2. Reducing the temporal resolution
3. Reducing the DSP frequency resolution

Downgrading these parameters reduces the computational load of feature extraction and shrinks spectrogram resolution, creating a subsequent reduction in downstream inference latency. However, it also reduces model accuracy.

<div align="center">

| Experiment | Parameter Adjusted | Processing Latency | Inference Latency | Total Latency | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | None (16KHz, 40 filters, 512 FFT, 0.01s stride) | 763 ms | 306 ms | 1,069 ms | 83.46% |
| **Downsampling** | Sample rate: 16KHz &rarr; 8KHz | 733 ms | 274 ms | 1,007 ms | 77.19% |
| **Time Resolution** | Frame stride: 0.01s &rarr; 0.02s | 381 ms | 100 ms | 481 ms | 76.60% |
| **DSP Resolution** | Filters: 40 &rarr; 20, FFT: 512 &rarr; 256 | 386 ms | 105 ms | **491 ms** | **79.20%** |

</div>

**Table 2.** *Impact of MFE processing block optimizations on system latency and test accuracy. See logs and screenshots in experiment_logs_and_outputs*

Downsampling to 8KHz provided a negligible latency improvement, while degrading accuracy. Conversely, decreasing the temporal resolution by doubling the frame stride successfully reduced total latency to below 500 ms at 481 ms, but had significantly reduced accuracy (76.60%).

Ultimately, the DSP resolution downgrade was selected as the optimal deployment configuration. By halving the number of mel-filters and the FFT length, the vertical frequency resolution of the spectrogram was compressed, achieving a total latency of 491 ms, while retaining a high test accuracy (79.20%).

### Real-World Validation

To evaluate the model on-device, a 150-shot data set including 50 noise, 50 miss and 50 make events was collected. While the model accuracy in this deployment test decreased to 67.3%, results showed that the sensor was robust to background noise with a 95.9% true negative rate (Figure 5). Furthermore, when the model detected a shot, it successfully distinguished between makes and misses, with an average misclassification rate of only 9%.

However, there was a significant false-negative rate, with over a third of valid shot events (36% of makes and 38% of misses) incorrectly classified as noise. This failure is primarily attributed to environmental domain shift. The training dataset was recorded over 3 windy afternoons, with one falling on the same day as the London Marathon, only 100 meters away. This likely embedded a high ambient noise floor so when the model was deployed on a quiet, low-wind morning, the fainter swish and rim sounds failed to overcome inference thresholds, causing the model to incorrectly default to noise classifications.

<div align="center">
  <img src="report_figures/deployment_test_confusion_matrix.png" alt="Deployment test confusion matrix" width="70%">
</div>

**Figure 5.** *Confusion matrix of the real-world classification performance across 150 acoustic events. Results are in experiment_logs_and_outputs/deployment_test.csv and were processed using experiment_logs_and_outputs/deployment_test.py in Google Collab.*

## Discussion and Conclusion

Swish successfully demonstrates TinyML’s viability for acoustic basketball shot tracking, achieving 79.2% test accuracy and 67.3% real-world deployment accuracy. The system effectively navigated latency challenges, featured a 95.9% background noise rejection rate and provided a user-friendly device with BLE stats retrieval and live LED feedback.

To evolve into a commercial sensor capable of tracking minute field-goal percentage changes, classification accuracy needs to be improved. Future model iterations will also need to be more robust and widely applicable as highlighted by the domain shift identified in deployment. To address this, a larger, more diverse dataset needs to be collected from a variety of environments and court set-ups (e.g. chain nets, indoor courts).

Another factor currently limiting the applicability of Swish is the inability for the model to detect 'airballs' and to distinguish shots on netless rims, due to a lack of impact signatures on these shot-types. Expanding the Swish system to include these shots may require multi-modal sensor fusion such as a rim-mounted accelerometer. 

Additionally, while device build provided adequate short-term durability as a prototype, a weatherproof, impact-resistant enclosure is necessary for longer-term use. Future iterations could also integrate a companion app, mapping shot statistics to court locations, elevating Swish from a shot tracker to a training tool.

Despite these limitations, this project provides a proof-of-concept to show that edge-based acoustic classification is a viable alternative to existing shot tracking technologies and demonstrates a user-friendly pipeline for its implementation.

## Bibliography
1. Cleary, T. J., & Zimmerman, B. J. (2001). Self-regulation differences during athletic practice by experts, non-experts, and novices. Journal of applied sport psychology, 13(2), 185-206.
2. Li, S., & Zhang, W. (2022). Evaluation Method of Basketball Teaching and training effect based on Wearable device. Frontiers in Physics, 10, 900169.
3. Lin, J., Chen, W. M., Lin, Y., Gan, C., & Han, S. (2020). Mcunet: Tiny deep learning on iot devices. Advances in neural information processing systems, 33, 11711-11722.
4. Moreira, P. E. D., Dieguez, G. T. D. O., Bredt, S. D. G. T., & Praça, G. M. (2021). The acute and chronic effects of dual-task on the motor and cognitive performances in athletes: a systematic review. International journal of environmental research and public health, 18(4), 1732.
5. Sandbrook, C., Luque-Lora, R., & Adams, W. M. (2018). Human bycatch: Conservation surveillance and the social implications of camera traps. Conservation and Society, 16(4), 493-504.

----

## Declaration of Authorship

I, Elliot Hills, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


*Elliot Hills*

05/05/2026 (DAP used)

Word count: 1489
