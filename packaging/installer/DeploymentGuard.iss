{ Refuse a live runtime before writing files. Restart Manager must not close a
  server/job or a busy Desktop automatically during a silent deployment. }
function DeploymentOpenFile(Name: String; Access, Sharing: Cardinal;
  Security: DWORD_PTR; Creation, Flags: Cardinal; Template: THandle): THandle;
  external 'CreateFileW@kernel32.dll stdcall';
function DeploymentCloseHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';

function DeploymentFilePath(Name: String): String;
begin
  if Copy(Name, 1, 4) = '\\?\' then Result := Name
  else if Copy(Name, 1, 2) = '\\' then Result := '\\?\UNC\' + Copy(Name, 3, MaxInt)
  else Result := '\\?\' + Name;
end;

function DeploymentBlockedFile(Directory: String): String;
var
  Found: TFindRec;
  Name, Extension: String;
  Handle: THandle;
begin
  Result := '';
  if not DirExists(Directory) then Exit;
  if FindFirst(AddBackslash(Directory) + '*', Found) then
  try
    repeat
      if (Found.Name <> '.') and (Found.Name <> '..') then
      begin
        Name := AddBackslash(Directory) + Found.Name;
        if (Found.Attributes and $400) <> 0 then
        begin
          Log('Deployment refused: reparse entry ' + Name);
          Result := Name;
          Exit;
        end;
        if (Found.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
          Result := DeploymentBlockedFile(Name)
        else
        begin
          Extension := Lowercase(ExtractFileExt(Name));
          if ((Extension = '.exe') or (Extension = '.dll') or (Extension = '.pyd')) and
             (CompareText(Copy(Found.Name, 1, 5), 'unins') <> 0) then
          begin
            { OPEN_EXISTING never changes bytes. A mapped image or another open
              handle denying writes/sharing makes the operation fail closed. }
            Handle := DeploymentOpenFile(DeploymentFilePath(Name), $40000000, 0, 0, 3, $80, 0);
            if Handle = THandle(-1) then
            begin
              Log('Deployment refused: busy/protected file ' + Name + ', error ' + IntToStr(DLLGetLastError));
              Result := Name;
            end
            else DeploymentCloseHandle(Handle);
          end;
        end;
        if Result <> '' then Exit;
      end;
    until not FindNext(Found);
  finally
    FindClose(Found);
  end;
end;

function DeploymentBlockedMessage(FileName: String): String;
begin
  Result := 'Kraken files are in use or protected: ' + FileName + #13#10 +
    'Close Kraken Desktop, stop Kraken Server and its jobs, then retry. ' +
    'The installer does not stop running applications automatically.';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Blocked: String;
begin
  Blocked := DeploymentBlockedFile(ExpandConstant('{app}'));
  Result := '';
  if Blocked <> '' then Result := DeploymentBlockedMessage(Blocked);
end;

function InitializeUninstall(): Boolean;
var
  Blocked: String;
begin
  Blocked := DeploymentBlockedFile(ExpandConstant('{app}'));
  Result := Blocked = '';
  if not Result then SuppressibleMsgBox(DeploymentBlockedMessage(Blocked), mbError, MB_OK, IDOK);
end;
