#pragma once

/*
 * ===========================================================================
 *  EDIT THIS ONE LINE, THEN UPLOAD.
 * ===========================================================================
 *
 *  Which node is this board?
 *
 *      1  = Node 1, Environment Safety   (MQ-135 + DHT11 + buzzer + LEDs)
 *      2  = Node 2, Kitchen / Occupancy  (DHT11 + PIR + sound sensor + LEDs)
 *      0  = MPU-6050 bench rig           (not used for the home system)
 *
 *  Change the number below, upload, and the board will stream the right
 *  columns for that node. Everything else is handled for you.
 */

#define SENSOR_PROFILE 1
