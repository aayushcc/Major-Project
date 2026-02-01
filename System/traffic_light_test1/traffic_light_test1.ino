// ---------------- PIN DEFINITIONS ----------------
const int red[4]    = {13, 27, 5, 15};
const int yellow[4] = {12, 26, 18, 2};
const int green[4]  = {14, 25, 19, 4};

String cmd = "";

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 4; i++) {
    pinMode(red[i], OUTPUT);
    pinMode(yellow[i], OUTPUT);
    pinMode(green[i], OUTPUT);
  }

  allRed();
  Serial.println("ESP32 READY");
}

// ---------------- LOOP ----------------
void loop() {
  if (Serial.available()) {
    cmd = Serial.readStringUntil('\n');
    cmd.trim();
    processCommand(cmd);
  }
}

// ---------------- FUNCTIONS ----------------

// Convert lane letter to index
int laneIndex(char lane) {
  if (lane == 'E') return 0;
  if (lane == 'S') return 1;
  if (lane == 'W') return 2;
  if (lane == 'N') return 3;
  return -1;
}

// Turn all lights RED
void allRed() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(red[i], HIGH);
    digitalWrite(yellow[i], LOW);
    digitalWrite(green[i], LOW);
  }
}

// Set a specific lane to a color
void setLane(int idx, char color) {
  if (idx < 0 || idx > 3) return;

  digitalWrite(red[idx], LOW);
  digitalWrite(yellow[idx], LOW);
  digitalWrite(green[idx], LOW);

  if (color == 'R') digitalWrite(red[idx], HIGH);
  if (color == 'Y') digitalWrite(yellow[idx], HIGH);
  if (color == 'G') digitalWrite(green[idx], HIGH);
}

// Process serial command
void processCommand(String cmd) {
  // Expected format: E,G,12
  if (cmd.length() < 5) return;

  char laneChar  = cmd.charAt(0);
  char colorChar = cmd.charAt(2);
  int duration   = cmd.substring(4).toInt();

  int idx = laneIndex(laneChar);
  if (idx == -1) return;

  setLane(idx, colorChar);

  Serial.print("Lane ");
  Serial.print(laneChar);
  Serial.print(" -> ");
  Serial.print(colorChar);
  Serial.print(" (");
  Serial.print(duration);
  Serial.println("s)");
}
