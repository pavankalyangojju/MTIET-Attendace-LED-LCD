import smbus
import time

# LCD Address
LCD_ADDRESS = 0x27  # Change to 0x3F if necessary

# Commands
LCD_CHR = 1  # Mode - Sending data
LCD_CMD = 0  # Mode - Sending command
LCD_LINE_1 = 0x80  # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0  # LCD RAM address for the 2nd line
LCD_BACKLIGHT = 0x08  # Backlight ON
ENABLE = 0b00000100  # Enable bit

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

class lcd:
    def __init__(self):
        self.bus = smbus.SMBus(1)  # I2C channel 1
        self.lcd_init()

    def lcd_init(self):
        """Initialize display."""
        self.lcd_byte(0x33, LCD_CMD)
        self.lcd_byte(0x32, LCD_CMD)
        self.lcd_byte(0x06, LCD_CMD)
        self.lcd_byte(0x0C, LCD_CMD)
        self.lcd_byte(0x28, LCD_CMD)
        self.lcd_byte(0x01, LCD_CMD)
        time.sleep(E_DELAY)

    def lcd_byte(self, bits, mode):
        """Send byte to data pins."""
        high_bits = mode | (bits & 0xF0) | LCD_BACKLIGHT
        low_bits = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
        self.bus.write_byte(LCD_ADDRESS, high_bits)
        self.lcd_toggle_enable(high_bits)
        self.bus.write_byte(LCD_ADDRESS, low_bits)
        self.lcd_toggle_enable(low_bits)

    def lcd_toggle_enable(self, bits):
        """Toggle enable pin."""
        time.sleep(E_DELAY)
        self.bus.write_byte(LCD_ADDRESS, bits | ENABLE)
        time.sleep(E_PULSE)
        self.bus.write_byte(LCD_ADDRESS, bits & ~ENABLE)
        time.sleep(E_DELAY)

    def lcd_clear(self):
        """Clear display."""
        self.lcd_byte(0x01, LCD_CMD)

    def lcd_display_string(self, message, line):
        """Display a string on the LCD."""
        self.lcd_byte(line, LCD_CMD)
        message = message.ljust(16, " ")
        for char in message:
            self.lcd_byte(ord(char), LCD_CHR)
