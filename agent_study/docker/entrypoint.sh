#!/bin/sh
# DockerWorkspace (openhands-workspace) hardcodes its docker run command to
# always append "--host 0.0.0.0 --port 8000" after the image name -- there's
# no field on DockerWorkspace to inject extra agent-server CLI flags. The
# client-side dynamic tool registration (RemoteConversation sending
# tool_module_qualnames at conversation-creation time) is documented in
# openhands/agent_server/__main__.py's own preload_modules() docstring as
# racy ("avoiding a race with dynamic tool_module_qualnames import") --
# confirmed live: importing TerminalTool/FileEditorTool client-side before
# constructing the Agent still produced a live
# KeyError: "ToolDefinition 'TerminalTool' is not registered" server-side.
# --import-modules is the documented, race-free alternative, so it's baked
# into the image's own entrypoint instead.
exec openhands-agent-server --import-modules openhands.tools.terminal,openhands.tools.file_editor "$@"
