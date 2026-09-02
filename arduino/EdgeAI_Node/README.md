# EdgeAI_Node - Arduino IDE sketch

**This folder is generated. Do not edit it by hand.**
It is produced from `firmware/` by `python tools/make_arduino_sketch.py`.
Edits here are lost the next time it is regenerated; change `firmware/` instead.

## Using it

1. Open `EdgeAI_Node.ino` in Arduino IDE.
2. Open the `node_select.h` tab and set `SENSOR_PROFILE` to the node you are
   flashing (1 = Environment Safety, 2 = Kitchen/Occupancy).
3. Board: **ESP32 Dev Module**. Upload speed 921600.
4. Install these libraries via Library Manager:
   - DHT sensor library (Adafruit)
   - Adafruit Unified Sensor
5. Upload, then open Serial Monitor at **115200 baud**.

The board prints a header line and then one CSV row per second.
