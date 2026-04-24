from model.edif import *
from pathlib import Path

def test_find_classes():
    test_text = '''(net N0020100
              (joined
		                (portRef DRAIN
                  (instanceRef INS28))
                (portRef DRAIN
                  (instanceRef INS30))
)
              (figure WIRE
                (path
                  (pointList 
                    (pt 3180 -1680) 
                    (pt 3180 -1660))))
                     )'''

    expected_result = ['''(joined
		                (portRef DRAIN
                  (instanceRef INS28))
                (portRef DRAIN
                  (instanceRef INS30))
)''']

    real_result = EdifParser.find_classes(test_text, '(joined')
    assert real_result == expected_result


def test_remove_header():
    test_text = '''(net N0020100
              (joined
		                (portRef DRAIN
                  (instanceRef INS28))
                (portRef DRAIN
                  (instanceRef INS30))
)
              (figure WIRE
                (path
                  (pointList 
                    (pt 3180 -1680) 
                    (pt 3180 -1660))))
                     )'''
    expected_text = '''(joined
		                (portRef DRAIN
                  (instanceRef INS28))
                (portRef DRAIN
                  (instanceRef INS30))
)
              (figure WIRE
                (path
                  (pointList 
                    (pt 3180 -1680) 
                    (pt 3180 -1660))))
                     )'''
    text_without_header = remove_edif_header(test_text)
    assert text_without_header == expected_text

def test_find_net_name():
    text_to_find = '''(net N0020100
              (joined
		                (portRef DRAIN
                  (instanceRef INS28))
                (portRef DRAIN
                  (instanceRef INS30))
)
              (figure WIRE
                (path
                  (pointList 
                    (pt 3180 -1680) 
                    (pt 3180 -1660))))
                     )'''

    expected_result = 'N0020100'
    real_result = find_net_name(text_to_find)
    assert real_result == expected_result

def test_extract_top_level_netlist():
    edif_path = Path("test_edifs/e1_model.EDF")
    parser = EdifParser(edif_path)
    netlist = parser.extract_top_level_netlist()

    assert netlist.design == "SCHEMATIC1"
    assert netlist.library == "CRPROJECT"
    assert netlist.cell == "SCHEMATIC1"
    assert netlist.view == "SCHEMATIC1_SCH"
    assert len(netlist.instances) > 0
    assert len(netlist.nets) > 0
    assert any(net.name == "N0020100" for net in netlist.nets)
    assert any(instance.x is not None and instance.y is not None for instance in netlist.instances)

def main():
    test_find_classes()
    test_remove_header()
    test_find_net_name()


if __name__ == '__main__':
  main()
