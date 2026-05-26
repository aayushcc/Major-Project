#include <WiFi.h>
#include <WebServer.h>

// ---------------- WIFI CONFIG ----------------
const char* ssid = "ESP32_TRAFFIC";
const char* password = "12345678";

WebServer server(80);

// ---------------- PIN DEFINITIONS ----------------
const int red[4]    = {14, 25, 5, 15};
const int yellow[4] = {12, 26, 18, 2};
const int green[4]  = {13, 27, 19, 4};

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);
  delay(2000);

  // Start Access Point
  WiFi.softAP(ssid, password);

  Serial.println("ESP32 WiFi Hotspot Started!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.softAPIP());

  // Setup pins
  for (int i = 0; i < 4; i++) {
    pinMode(red[i], OUTPUT);
    pinMode(yellow[i], OUTPUT);
    pinMode(green[i], OUTPUT);
  }

  allRed();

  // Define route
  server.on("/set", handleSet);

  server.begin();
  Serial.println("HTTP server started");
}

// ---------------- LOOP ----------------
void loop() {
  server.handleClient();
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

// Set lane color
void setLane(int idx, char color) {
  if (idx < 0 || idx > 3) return;

  digitalWrite(red[idx], LOW);
  digitalWrite(yellow[idx], LOW);
  digitalWrite(green[idx], LOW);

  if (color == 'R') digitalWrite(red[idx], HIGH);
  if (color == 'Y') digitalWrite(yellow[idx], HIGH);
  if (color == 'G') digitalWrite(green[idx], HIGH);
}

// Handle HTTP request
void handleSet() {
  if (!server.hasArg("lane") || !server.hasArg("color")) {
    server.send(400, "text/plain", "Missing parameters");
    return;
  }

  String laneStr = server.arg("lane");
  String colorStr = server.arg("color");
  int duration = server.hasArg("time") ? server.arg("time").toInt() : 0;

  char laneChar = laneStr.charAt(0);
  char colorChar = colorStr.charAt(0);

  int idx = laneIndex(laneChar);

  if (idx == -1) {
    server.send(400, "text/plain", "Invalid lane");
    return;
  }

  setLane(idx, colorChar);

  Serial.print("Lane ");
  Serial.print(laneChar);
  Serial.print(" -> ");
  Serial.print(colorChar);
  Serial.print(" (");
  Serial.print(duration);
  Serial.println("s)");

  server.send(200, "text/plain", "OK");
}
