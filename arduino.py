import serial
import time
import config

def initialize_arduino(ARDUINO_DEVICE, baudrate=9600, delay=2):
    """
    Attempts to initialize connection with the Arduino over serial.
    
    Args:
        ARDUINO_DEVICE (str): Serial port (e.g., '/dev/ttyUSB0' or 'COM3').
        baudrate (int): Baud rate for serial communication.
        delay (int): Seconds to wait after opening the port.
    """
    try:
        config.arduino = serial.Serial(ARDUINO_DEVICE, baudrate)
        time.sleep(delay)  # Allow time for Arduino to reset
        print(f"[Arduino] Connected on {ARDUINO_DEVICE} at {baudrate} baud.")
    except Exception as e:
        print(f"[Arduino] Failed to connect: {e}")
        config.arduino = None


def send_pwm_value(value):
    """
    Sends a PWM value to the Arduino via serial.
    
    Args:
        value (int): Value between 0 and 255 to be sent.
    """
    if config.arduino and config.arduino.is_open:
        pwm = int(max(0, min(255, value)))  # Clamp value between 0 and 255
        try:
            config.arduino.write(f"{pwm}\n".encode())
        except Exception as e:
            print(f"[Arduino] Write error: {e}")
