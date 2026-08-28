import unittest

from src.motor import build_motor_commands, level_to_pwm, run_motor


class MotorTests(unittest.TestCase):
    def test_level_conversion(self):
        self.assertEqual(level_to_pwm(0), 1000)
        self.assertEqual(level_to_pwm(3), 1300)
        self.assertEqual(level_to_pwm(10), 2000)

    def test_commands_contain_four_motor_values(self):
        commands = build_motor_commands((0, 0, 0, 3))
        self.assertEqual(commands, (b"motor 0 1000\r\n", b"motor 1 1000\r\n", b"motor 2 1000\r\n", b"motor 3 1300\r\n"))

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(ValueError):
            level_to_pwm(11)

    def test_run_one_motor(self):
        class FakeSerial:
            def __init__(self):
                self.commands = []

            def write(self, command):
                self.commands.append(command)

            def flush(self):
                pass

        serial_port = FakeSerial()
        command = run_motor(serial_port, 3, 1)
        self.assertEqual(command, b"motor 3 1100\r\n")
        self.assertEqual(serial_port.commands, [b"motor 3 1100\r\n"])

    def test_invalid_motor_number_is_rejected(self):
        with self.assertRaises(ValueError):
            run_motor(object(), 4, 1)


if __name__ == "__main__":
    unittest.main()
