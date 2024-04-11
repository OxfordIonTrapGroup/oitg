import unittest
from oitg.results import *


class MagicTest(unittest.TestCase):
    def test_parse(self):
        def t(string, expected):
            self.assertEqual(parse_magic(string), expected)

        t("1", {"rid": 1})
        t("0123", {"rid": 123})
        t("rid_123", {"rid": 123})
        t("comet_123", {"rid": 123, "experiment": "comet"})
        t("alice_2", {"rid": 2, "experiment": "lab1_alice"})
        t("bob_123", {"rid": 123, "experiment": "lab1_bob"})
        t("lab1_blade_123", {"rid": 123, "experiment": "lab1_blade"})
