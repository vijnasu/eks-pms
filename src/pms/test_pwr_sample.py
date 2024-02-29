import unittest
from pwr_sample import MyClass

class TestMyClass(unittest.TestCase):
    def setUp(self):
        self.my_obj = MyClass()

    def test_batch_adjust_configurations(self):
        # Test case 1: Adjusting uncore frequencies
        self.my_obj.is_uncore = True
        self.my_obj.units = [Unit(uncore_kernel_avail=True, uncore_min_freq=100, uncore_max_freq=200)]
        self.my_obj.validate_frequency = lambda freq: True  # Mocking the validate_frequency method
        self.my_obj.display_current_configurations = lambda: None  # Mocking the display_current_configurations method
        self.my_obj.console = MockConsole()  # Mocking the console object

        self.my_obj.batch_adjust_configurations()

        self.assertEqual(self.my_obj.units[0].uncore_min_freq, 100)  # Check if min_freq is unchanged
        self.assertEqual(self.my_obj.units[0].uncore_max_freq, 200)  # Check if max_freq is unchanged

        # Test case 2: Adjusting core frequencies
        self.my_obj.is_uncore = False
        self.my_obj.units = [Unit(online=True, min_freq=100, max_freq=200)]
        self.my_obj.validate_frequency = lambda freq: True  # Mocking the validate_frequency method
        self.my_obj.display_current_configurations = lambda: None  # Mocking the display_current_configurations method
        self.my_obj.console = MockConsole()  # Mocking the console object

        self.my_obj.batch_adjust_configurations()

        self.assertEqual(self.my_obj.units[0].min_freq, 100)  # Check if min_freq is unchanged
        self.assertEqual(self.my_obj.units[0].max_freq, 200)  # Check if max_freq is unchanged

        # Test case 3: Adjusting uncore frequencies with out-of-range input
        self.my_obj.is_uncore = True
        self.my_obj.units = [Unit(uncore_kernel_avail=True, uncore_min_freq=100, uncore_max_freq=200)]
        self.my_obj.validate_frequency = lambda freq: False  # Mocking the validate_frequency method
        self.my_obj.display_current_configurations = lambda: None  # Mocking the display_current_configurations method
        self.my_obj.console = MockConsole()  # Mocking the console object

        self.my_obj.batch_adjust_configurations()

        self.assertEqual(self.my_obj.units[0].uncore_min_freq, 100)  # Check if min_freq is unchanged
        self.assertEqual(self.my_obj.units[0].uncore_max_freq, 200)  # Check if max_freq is unchanged

if __name__ == '__main__':
    unittest.main()