import unittest

import nb_runner

class TestNotebook(unittest.TestCase):

    def test_runner(self):
        nb, errors = nb_runner.run_notebook('Exercise_1_01.ipynb')
        self.assertEqual(errors, [])

if __name__ == '__main__':
    unittest.main()