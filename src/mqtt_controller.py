import paho.mqtt.client as mqtt


class MQTTController:
    """
    Handles MQTT communication for the face-centering system.

    Week 1: horizontal (pan) camera movement.
    Week 2: vertical (tilt) camera movement added.
    """

    BROKER = "127.0.0.1"
    PORT = 1883

    TOPIC_HORIZONTAL = "face_centering/motor/horizontal"
    TOPIC_VERTICAL = "face_centering/motor/vertical"

    # direction (from the tracker) -> command (for the ESP8266)
    PAN_COMMANDS = {
        "LEFT": "PAN_LEFT",
        "RIGHT": "PAN_RIGHT",
        "CENTER": "PAN_STOP",
    }

    TILT_COMMANDS = {
        "UP": "TILT_UP",
        "DOWN": "TILT_DOWN",
        "CENTER": "TILT_STOP",
    }

    def __init__(self):
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="face_centering_pc",
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self.connected = False

        # last command sent per topic, so we only publish on change
        self.last_command = {
            self.TOPIC_HORIZONTAL: None,
            self.TOPIC_VERTICAL: None,
        }

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"[MQTT] Connected to broker: {reason_code}")
        self.connected = True

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ):
        self.connected = False
        print(f"[MQTT] Disconnected: {reason_code}")

    def connect(self):
        print(f"[MQTT] Connecting to {self.BROKER}:{self.PORT}...")
        self.client.connect(self.BROKER, self.PORT, keepalive=60)
        # Paho's network loop runs in a background thread.
        self.client.loop_start()

    def _publish(self, topic, command):
        """Publish only if the command changed since last time."""
        if command == self.last_command[topic]:
            return

        if not self.connected:
            print(f"[MQTT] Not connected. Could not send {command}")
            return

        result = self.client.publish(topic, command, qos=0, retain=False)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] {topic} -> {command}")
            self.last_command[topic] = command
        else:
            print(f"[MQTT] Publish failed: {result.rc}")

    def publish_horizontal(self, direction):
        command = self.PAN_COMMANDS.get(direction)
        if command is None:
            print(f"[MQTT] Unknown horizontal direction: {direction}")
            return
        self._publish(self.TOPIC_HORIZONTAL, command)

    def publish_vertical(self, direction):
        command = self.TILT_COMMANDS.get(direction)
        if command is None:
            print(f"[MQTT] Unknown vertical direction: {direction}")
            return
        self._publish(self.TOPIC_VERTICAL, command)

    def publish_tracking(self, horizontal, vertical):
        """Send both axes from one TrackingSignal."""
        self.publish_horizontal(horizontal)
        self.publish_vertical(vertical)

    def stop(self):
        """Immediately stop both motors (e.g. face lost)."""
        self._publish(self.TOPIC_HORIZONTAL, "PAN_STOP")
        self._publish(self.TOPIC_VERTICAL, "TILT_STOP")

    def disconnect(self):
        self.stop()
        self.client.loop_stop()
        self.client.disconnect()
        print("[MQTT] Disconnected")