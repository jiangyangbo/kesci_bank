import unittest

from process import comm
from logger import logger



class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.accept_months = []
        self.accept_months.append((2014, 8))
        self.accept_months.append((2015, 7))
        self.accept_months.append((2014, 10))

    def test_accept(self):
        test_data = []
        test_data.append('2014-08-15 10:27:36')
        test_data.append('2014-08-15 10:27:36')
        test_data.append('2015-7-15 10:27:36')
        test_data.append('2014-10-15 10:27:36')
        for d in test_data:
            logger.info(d)
            r = comm.accept_this_date(d, self.accept_months)
            self.assertEqual(r, True)

    def test_not_accept(self):
        test_data = []
        test_data.append('2014-06-15 10:27:36')
        test_data.append('2014-02-15 10:27:36')
        test_data.append('2015-08-15 10:27:36')
        test_data.append('2014-11-15 10:27:36')
        for d in test_data:
            logger.info(d)
            r = comm.accept_this_date(d, self.accept_months)
            self.assertEqual(r, False)


if __name__ == '__main__':
    unittest.main()
