import time
import keyboard
import pyautogui

SCROLL_AMOUNT = -60   # negative = scroll down, positive = scroll up
SCROLL_DELAY = 0.00  # seconds between scrolls

scrolling = False  # toggle state

print("Press 'd' to toggle scrolling. Press 'q' to quit.")

try:
    while True:
        if keyboard.is_pressed('q'):
            break

        # Toggle scrolling when 'd' is pressed
        if keyboard.is_pressed(' '):
            scrolling = not scrolling
            print("Scrolling ON" if scrolling else "Scrolling OFF")
            while keyboard.is_pressed('d'):  # wait for release
                time.sleep(0.05)

        # Perform scrolling if enabled
        if scrolling:
            pyautogui.scroll(SCROLL_AMOUNT)
            time.sleep(SCROLL_DELAY)
        else:
            time.sleep(0.01)

except KeyboardInterrupt:
    pass
