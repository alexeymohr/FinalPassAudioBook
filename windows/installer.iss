; Inno Setup script: one-click, per-user Windows installer for FinalPass AudioBook.
; Build with:  ISCC.exe /DAppVersion=<version> installer.iss
#define AppName "FinalPass AudioBook"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#define AppExe "fpab-gui.exe"

[Setup]
AppId={{6F3A2B14-7C5D-4E8A-9B21-0D4E6F8A1C33}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Alexey Mohr
DefaultDirName={localappdata}\Programs\FinalPass AudioBook
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=dist
OutputBaseFilename=FinalPassAudioBook-{#AppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
InfoBeforeFile=EXPERIMENTAL.txt
; SetupIconFile=app.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\FinalPassAudioBook\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "staging\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
