from mqtt_controller import MQTTController


mqtt = MQTTController()

try:
    mqtt.connect()

    print()
    print("MQTT TEST")
    print("Type:")
    print("  l = LEFT")
    print("  r = RIGHT")
    print("  s = STOP")
    print("  q = quit")
    print()

    while True:

        command = input("> ").strip().lower()

        if command == "l":
            mqtt.publish_horizontal("LEFT")

        elif command == "r":
            mqtt.publish_horizontal("RIGHT")

        elif command == "s":
            mqtt.publish_horizontal("CENTER")

        elif command == "q":
            break

        else:
            print("Unknown command.")

finally:
    mqtt.disconnect()