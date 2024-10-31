import unittest
from sklearn.isotonic import check_increasing
import pandas as pd 


path = "./test_calibration_data.csv"
test_data = pd.read_csv(path)


class TestCalirate(unittest.TestCase):

    def test_monotonic(self):
        condition = check_increasing(test_data.proba_predict, test_data.proba_calibration)
        self.assertTrue(condition)
    


if __name__ == "__main__":
    unittest.main()

    