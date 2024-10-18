import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class TestNotebooks(unittest.TestCase):
    def test_adsorbate_distribution(self):
        with open('Examples/adsorbate_distribution.ipynb') as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': 'Examples/'}})

    def test_layer_descriptor(self):
        with open('Examples/layer_descriptor.ipynb') as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': 'Examples/'}})

    # Add similar methods for other notebooks

if __name__ == '__main__':
    unittest.main()
