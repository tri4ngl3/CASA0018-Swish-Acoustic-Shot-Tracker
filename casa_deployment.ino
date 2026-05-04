#include <WXWX9_CASA0018_PROJECT_inferencing.h> 
#include <ArduinoBLE.h>
#include <PDM.h>

BLEService uartService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
BLEStringCharacteristic txChar("6E400003-B5A3-F393-E0A9-E50E24DCCA9E", BLERead | BLENotify, 128); 
BLEStringCharacteristic rxChar("6E400002-B5A3-F393-E0A9-E50E24DCCA9E", BLEWrite | BLEWriteWithoutResponse, 20);

int makeCount = 0;
int missCount = 0;
unsigned long lastShotTime = 0;
const unsigned long COOLDOWN_MS = 2000; 

typedef struct {
    int16_t *buffer;
    uint8_t buf_ready;
    uint32_t buf_count;
    uint32_t n_samples;
} inference_t;

static inference_t inference;
static signed short sampleBuffer[2048];
static bool debug_nn = false;

void setLED(bool red, bool green, bool blue) {
    digitalWrite(LEDR, red ? LOW : HIGH);
    digitalWrite(LEDG, green ? LOW : HIGH);
    digitalWrite(LEDB, blue ? LOW : HIGH);
}

void setup() {
    Serial.begin(115200);
    
    pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
    setLED(false, false, true); 

    if (!BLE.begin()) {
        Serial.println("Starting BLE failed!");
        while (1);
    }
    
    BLE.setLocalName("HoopTracker");
    BLE.setAdvertisedService(uartService);
    uartService.addCharacteristic(txChar);
    uartService.addCharacteristic(rxChar);
    BLE.addService(uartService);
    BLE.advertise();
    Serial.println("UART BLE Active. Waiting for texts...");

    if (EI_CLASSIFIER_RAW_SAMPLES_PER_FRAME != 1) {
        Serial.println("ERR: Audio must be mono"); while (1);
    }
    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
        Serial.println("ERR: Failed to setup audio"); while (1);
    }
}

void loop() {
    BLE.poll();

    if (rxChar.written()) {
        String msg = rxChar.value();
        msg.trim(); msg.toUpperCase();
        
        if (msg == "REPORT") {
            int totalShots = makeCount + missCount;
            float fgPercent = 0.0;
            if (totalShots > 0) fgPercent = ((float)makeCount / (float)totalShots) * 100.0;
            
            String report = "SESSION OVERVIEW:\nTook: " + String(totalShots) + " shots\nMade: " + String(makeCount) + "\nMissed: " + String(missCount) + "\nFG%: " + String(fgPercent, 1) + "%";
            txChar.writeValue(report);
            Serial.println("Report sent to phone.");
        } 
        else if (msg == "RESET") {
            makeCount = 0; missCount = 0;
            txChar.writeValue("Counters reset to 0. Have a good session!");
        }
    }

    bool m = microphone_inference_record();
    if (!m) return;

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};
    EI_IMPULSE_ERROR r = run_classifier(&signal, &result, debug_nn);
    if (r != EI_IMPULSE_OK) return;

    float maxConfidence = 0.0;
    String bestLabel = "";

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        if (result.classification[i].value > maxConfidence) {
            maxConfidence = result.classification[i].value;
            bestLabel = String(result.classification[i].label);
        }
    }

    bestLabel.trim();

    if (maxConfidence > 0.70) {
        unsigned long currentTime = millis();
        bool cooldownPassed = (currentTime - lastShotTime) > COOLDOWN_MS;

        if (bestLabel.equalsIgnoreCase("Make") && cooldownPassed) {
            makeCount++; lastShotTime = currentTime; setLED(false, true, false); 
            Serial.println("SWISH!");
        } 
        else if (bestLabel.equalsIgnoreCase("Miss") && cooldownPassed) {
            missCount++; lastShotTime = currentTime; setLED(true, false, false); 
            Serial.println("BRICK!");
        }
        else if (bestLabel.equalsIgnoreCase("Noise") && cooldownPassed) {
            setLED(false, false, true); 
        }
    }
}

static void pdm_data_ready_inference_callback(void) {
    int bytesAvailable = PDM.available();
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);
    if (inference.buf_ready == 0) {
        for(int i = 0; i < bytesRead>>1; i++) {
            inference.buffer[inference.buf_count++] = sampleBuffer[i];
            if(inference.buf_count >= inference.n_samples) {
                inference.buf_count = 0; inference.buf_ready = 1; break;
            }
        }
    }
}
static bool microphone_inference_start(uint32_t n_samples) {
    inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if(inference.buffer == NULL) return false;
    inference.buf_count = 0; inference.n_samples = n_samples; inference.buf_ready = 0;
    PDM.onReceive(&pdm_data_ready_inference_callback);
    PDM.setBufferSize(4096);
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) return false;
    PDM.setGain(127); return true;
}
static bool microphone_inference_record(void) {
    inference.buf_ready = 0; inference.buf_count = 0;
    while(inference.buf_ready == 0) delay(10);
    return true;
}
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length); return 0;
}