void setup()
{
  Serial.begin(9600); // Bluetooth modules HC-05, use 9600 baud
  pinMode(9, OUTPUT); // Connect motor to pin 9 (PWM)
}

void loop()
{
  if (Serial.available())
  {
    int val = Serial.parseInt();  // Read the incoming vibration value (0–255)
    val = constrain(val, 0, 255); // Ensure it's within bounds
    analogWrite(9, val);          // Send to vibration motor
    delay(5);                     // Small delay to prevent overflow
  }
}