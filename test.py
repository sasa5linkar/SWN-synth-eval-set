import unittest
import os
from dotenv import load_dotenv
from srpskiwn import srpskiwordnet


class TestFilePath(unittest.TestCase):
    def test_file_path(self):
        # Load .env file
        load_dotenv()

        # Retrieve file root and file name
        file_root = os.getenv('srpwn_root')
        file_name = os.getenv('srpwn_file')

        # Join root and file name to form complete path
        file_path = os.path.join(file_root, file_name)

        # Print file path
        print(file_path)

        # Check if file exists
        self.assertTrue(os.path.exists(file_path), "File does not exist")

class TestWN(unittest.TestCase):
    def test_synonyms(self):
        # Load .env file
        load_dotenv()

        # Retrieve file root and file name
        file_root = os.getenv('srpwn_root')
        file_name = os.getenv('srpwn_file')


        # Create srpskiwordnet object
        wn = srpskiwordnet.SrbWordNetReader(file_root, file_name)

        # Retrieve synonyms
        synonyms = wn.synsets('lep')

        # Print synonyms
        print(synonyms)


        # Check if synonyms are not empty
        self.assertTrue(len(synonyms) > 0, "Synonyms are empty")


if __name__ == '__main__':
    unittest.main()