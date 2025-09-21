from lxml import etree


def is_well_formed(xml_text):
try:
etree.fromstring(f"<root>\n{xml_text}\n</root>")
return True, None
except Exception as e:
return False, str(e)

if __name__ == '__main__':
import sys
txt = open(sys.argv[1]).read()
ok, err = is_well_formed(txt)
if ok:
print('OK: well-formed XML (wrapped in <root>).')
else:
print('NOT OK:', err)