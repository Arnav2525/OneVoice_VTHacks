' Double-click to start OneVoice with no console windows. Add "live" for the hardware demo.
Set shell = CreateObject("WScript.Shell")
here = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
args = ""
For Each a In WScript.Arguments
  args = args & " " & a
Next
shell.Run Chr(34) & here & "\Start OneVoice Experience.cmd" & Chr(34) & args, 0, False
