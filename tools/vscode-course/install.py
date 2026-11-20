"""Build the course.py helper as a .vsix and install it into VS Code (and Cursor, if present).

    python3 tools/vscode-course/install.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("package.json", "extension.js", "docs.json", "check_course.py")
EDITORS = ("code", "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
           "/Applications/Cursor.app/Contents/Resources/app/bin/cursor")


def build(out):
    pkg = json.loads((HERE / "package.json").read_text(encoding="utf-8"))
    manifest = f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="{pkg['name']}" Version="{pkg['version']}" Publisher="{pkg['publisher']}" />
    <DisplayName>{pkg['displayName']}</DisplayName>
    <Description xml:space="preserve">{pkg['description']}</Description>
    <Categories>Other</Categories>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{pkg['engines']['vscode']}" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace" />
    </Properties>
  </Metadata>
  <Installation><InstallationTarget Id="Microsoft.VisualStudio.Code" /></Installation>
  <Dependencies />
  <Assets><Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" /></Assets>
</PackageManifest>
"""
    types = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension=".json" ContentType="application/json" /><Default Extension=".js" ContentType="application/javascript" />
<Default Extension=".py" ContentType="text/plain" /><Default Extension=".vsixmanifest" ContentType="text/xml" />
</Types>
"""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("extension.vsixmanifest", manifest)
        for f in FILES:
            z.write(HERE / f, f"extension/{f}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        vsix = Path(tmp) / "aicar-course.vsix"
        build(vsix)
        done = set()
        for ed in EDITORS:
            exe = shutil.which(ed) or (ed if Path(ed).exists() else None)
            if not exe or Path(exe).name in done:
                continue
            done.add(Path(exe).name)
            r = subprocess.run([exe, "--install-extension", str(vsix), "--force"], capture_output=True, text=True)
            print(f"{Path(exe).name}: {'installed' if r.returncode == 0 else 'failed: ' + (r.stderr or r.stdout).strip()}")
        if not done:
            sys.exit("VS Code not found")
    print("Reload the editor window (Developer: Reload Window) to use it.")


if __name__ == "__main__":
    main()
