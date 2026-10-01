  /*
    Drowsiness Detection Alert System
    ==================================
    Receives alert level from Python via Serial:
      0 = Normal   → Green LED on
      1 = Warning  → Yellow LED on + slow beep
      2 = Critical → Red LEDs flash + rapid beep

    Minimum duration enforcement:
      Warning stays active for at least 1 second before
      escalating to Critical, and at least 2 seconds before
      returning to Normal; Critical stays at least 1 second.
      This ensures each LED state is visible. A state equal to
      the displayed one cancels any queued change.

    Hardware:
      Pin 13 = Green  LED  (Normal)
      Pin 12 = Yellow LED  (Warning)
      Pin 11 = Red    LED  (Critical solid)
      Pin 10 = Red    LED  (Critical flash)
      Pin  9 = Buzzer      (Active buzzer)
  */

  // ── Pin definitions ─────────────────────────────────────────
  const int PIN_LED_GREEN  = 13;
  const int PIN_LED_YELLOW = 12;
  const int PIN_LED_RED1   = 11;
  const int PIN_LED_RED2   = 10;
  const int PIN_BUZZER     = 9;

  // ── Alert states ─────────────────────────────────────────────
  const int STATE_NORMAL   = 0;
  const int STATE_WARNING  = 1;
  const int STATE_CRITICAL = 2;

  // ── Timing ───────────────────────────────────────────────────
  // Warning beep pattern
  const unsigned long WARNING_BEEP_ON    = 300;    // ms
  const unsigned long WARNING_BEEP_OFF   = 1700;   // ms

  // Critical beep pattern
  const unsigned long CRITICAL_BEEP_ON   = 100;    // ms
  const unsigned long CRITICAL_BEEP_OFF  = 100;    // ms
  const unsigned long CRITICAL_FLASH_MS  = 150;    // ms

  // Minimum time each state must be active
  // Prevents state from changing too fast
  const unsigned long MIN_WARNING_DURATION  = 2000; // 2 seconds
  const unsigned long MIN_CRITICAL_DURATION = 1000; // 1 second

  // ── State variables ──────────────────────────────────────────
  int  currentState       = STATE_NORMAL;
  int  pendingState       = STATE_NORMAL;
  bool stateChangePending = false;

  // Timing for non-blocking blink/beep
  unsigned long stateEnteredAt     = 0;
  unsigned long lastBuzzerToggle   = 0;
  unsigned long lastLEDToggle      = 0;
  bool          buzzerOn           = false;
  bool          led2On             = false;

  // Serial input buffer
  String inputBuffer = "";

  // Heartbeat - blink green LED to show Arduino is alive
  unsigned long lastHeartbeat = 0;


  // ═════════════════════════════════════════════════════════════
  // SETUP
  // ═════════════════════════════════════════════════════════════

  void setup() {
    pinMode(PIN_LED_GREEN,  OUTPUT);
    pinMode(PIN_LED_YELLOW, OUTPUT);
    pinMode(PIN_LED_RED1,   OUTPUT);
    pinMode(PIN_LED_RED2,   OUTPUT);
    pinMode(PIN_BUZZER,     OUTPUT);

    allOff();

    Serial.begin(9600);

    startupTest();

    stateEnteredAt = millis();
    Serial.println("READY");
  }


  // ═════════════════════════════════════════════════════════════
  // LOOP
  // ═════════════════════════════════════════════════════════════

  void loop() {
    // Read serial from Python
    readSerial();

    // Handle pending state changes with minimum duration
    handlePendingStateChange();

    // Run current alert pattern
    switch (currentState) {
      case STATE_NORMAL:
        runNormal();
        break;
      case STATE_WARNING:
        runWarning();
        break;
      case STATE_CRITICAL:
        runCritical();
        break;
    }
  }


  // ═════════════════════════════════════════════════════════════
  // SERIAL READER
  // ═════════════════════════════════════════════════════════════

  void readSerial() {
    while (Serial.available() > 0) {
      char c = (char)Serial.read();

      if (c == '\n') {
        inputBuffer.trim();

        if (inputBuffer.length() > 0) {
          int newState = inputBuffer.toInt();

          if (newState >= STATE_NORMAL &&
              newState <= STATE_CRITICAL) {

            if (newState != currentState) {
              // Don't change immediately
              // Queue it for minimum duration check
              pendingState       = newState;
              stateChangePending = true;
            } else {
              // PC is back in the displayed state: cancel any queued
              // change (otherwise a queued Critical would still fire
              // after the eyes had reopened)
              stateChangePending = false;
            }
          }
        }
        inputBuffer = "";

      } else if (inputBuffer.length() < 8) {
        // Commands are one digit; ignore runaway input without a newline
        inputBuffer += c;
      }
    }
  }


  // ═════════════════════════════════════════════════════════════
  // PENDING STATE CHANGE HANDLER
  // Enforces minimum duration for each state
  // ═════════════════════════════════════════════════════════════

  void handlePendingStateChange() {
    if (!stateChangePending) {
      return;
    }

    unsigned long now      = millis();
    unsigned long duration = now - stateEnteredAt;

    // Check if current state has been active long enough
    bool canChange = false;

    switch (currentState) {
      case STATE_NORMAL:
        // Normal can always change (no minimum)
        canChange = true;
        break;

      case STATE_WARNING:
        // Warning must stay for at least 2 seconds
        // UNLESS escalating to Critical (always allow)
        if (pendingState == STATE_CRITICAL) {
          canChange = (duration >= MIN_CRITICAL_DURATION);
        } else {
          canChange = (duration >= MIN_WARNING_DURATION);
        }
        break;

      case STATE_CRITICAL:
        // Critical must stay for at least 1 second
        canChange = (duration >= MIN_CRITICAL_DURATION);
        break;
    }

    if (canChange) {
      // Apply the state change
      currentState       = pendingState;
      stateChangePending = false;
      onStateChange();
    }
  }


  // ═════════════════════════════════════════════════════════════
  // STATE CHANGE HANDLER
  // ═════════════════════════════════════════════════════════════

  void onStateChange() {
    allOff();

    // Reset timing
    stateEnteredAt   = millis();
    lastBuzzerToggle = millis();
    lastLEDToggle    = millis();
    buzzerOn         = false;
    led2On           = false;

    // Send acknowledgment to Python
    Serial.print("STATE:");
    Serial.println(currentState);
  }


  // ═════════════════════════════════════════════════════════════
  // STATE: NORMAL (0)
  // Green LED on, silent
  // ═════════════════════════════════════════════════════════════

  void runNormal() {
    digitalWrite(PIN_LED_GREEN,  HIGH);
    digitalWrite(PIN_LED_YELLOW, LOW);
    digitalWrite(PIN_LED_RED1,   LOW);
    digitalWrite(PIN_LED_RED2,   LOW);
    digitalWrite(PIN_BUZZER,     LOW);
  }


  // ═════════════════════════════════════════════════════════════
  // STATE: WARNING (1)
  // Yellow LED on
  // Slow beep: 300ms on, 1700ms off (once every 2 seconds)
  // ═════════════════════════════════════════════════════════════

  void runWarning() {
    digitalWrite(PIN_LED_GREEN,  LOW);
    digitalWrite(PIN_LED_YELLOW, HIGH);
    digitalWrite(PIN_LED_RED1,   LOW);
    digitalWrite(PIN_LED_RED2,   LOW);

    unsigned long now = millis();

    if (buzzerOn) {
      if (now - lastBuzzerToggle >= WARNING_BEEP_ON) {
        digitalWrite(PIN_BUZZER, LOW);
        buzzerOn         = false;
        lastBuzzerToggle = now;
      }
    } else {
      if (now - lastBuzzerToggle >= WARNING_BEEP_OFF) {
        digitalWrite(PIN_BUZZER, HIGH);
        buzzerOn         = true;
        lastBuzzerToggle = now;
      }
    }
  }


  // ═════════════════════════════════════════════════════════════
  // STATE: CRITICAL (2)
  // Red LEDs flash rapidly
  // Rapid beep: 100ms on, 100ms off
  // ═════════════════════════════════════════════════════════════

  void runCritical() {
    digitalWrite(PIN_LED_GREEN,  LOW);
    digitalWrite(PIN_LED_YELLOW, LOW);
    digitalWrite(PIN_LED_RED1,   HIGH);  // Solid red

    unsigned long now = millis();

    // Flash second red LED
    if (now - lastLEDToggle >= CRITICAL_FLASH_MS) {
      led2On       = !led2On;
      lastLEDToggle = now;
      digitalWrite(PIN_LED_RED2, led2On ? HIGH : LOW);
    }

    // Rapid buzzer
    if (now - lastBuzzerToggle >= CRITICAL_BEEP_ON) {
      buzzerOn         = !buzzerOn;
      lastBuzzerToggle = now;
      digitalWrite(PIN_BUZZER, buzzerOn ? HIGH : LOW);
    }
  }


  // ═════════════════════════════════════════════════════════════
  // HELPERS
  // ═════════════════════════════════════════════════════════════

  void allOff() {
    digitalWrite(PIN_LED_GREEN,  LOW);
    digitalWrite(PIN_LED_YELLOW, LOW);
    digitalWrite(PIN_LED_RED1,   LOW);
    digitalWrite(PIN_LED_RED2,   LOW);
    digitalWrite(PIN_BUZZER,     LOW);
  }


  void startupTest() {
    /*
      Runs once at power on.
      Each LED lights up for 500ms.
      Buzzer beeps once.
      Confirms all components work.
    */
    Serial.println("Testing components...");

    // Green
    digitalWrite(PIN_LED_GREEN, HIGH);
    delay(500);
    digitalWrite(PIN_LED_GREEN, LOW);

    // Yellow
    digitalWrite(PIN_LED_YELLOW, HIGH);
    delay(500);
    digitalWrite(PIN_LED_YELLOW, LOW);

    // Red 1
    digitalWrite(PIN_LED_RED1, HIGH);
    delay(500);
    digitalWrite(PIN_LED_RED1, LOW);

    // Red 2
    digitalWrite(PIN_LED_RED2, HIGH);
    delay(500);
    digitalWrite(PIN_LED_RED2, LOW);

    // Buzzer
    digitalWrite(PIN_BUZZER, HIGH);
    delay(300);
    digitalWrite(PIN_BUZZER, LOW);

    Serial.println("Component test done");
    delay(300);
  }