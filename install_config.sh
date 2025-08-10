#!/bin/bash

# TPM Assistant - Claude Desktop Configuration Installer
# This script installs the MCP server configuration for Claude Desktop

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_CONFIG_DIR="$HOME/.config/Claude"
CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"

echo "🚀 Installing TPM Assistant configuration for Claude Desktop..."

# Create Claude config directory if it doesn't exist
mkdir -p "$CLAUDE_CONFIG_DIR"

# Create the configuration with current project directory
cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "tpm-assistant": {
      "command": "$PROJECT_DIR/start_mcp.sh",
      "args": [],
      "env": {
        "PATH": "\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  },
  "globalShortcuts": {
    "tpm-welcome": "claude_desktop_welcome",
    "tmp-discover": "discover_tpm_capabilities", 
    "tpm-help": "tpm_help_and_examples",
    "tpm-data": "get_sample_data_overview"
  },
  "autoRun": {
    "onConnect": ["claude_desktop_welcome"],
    "tools": {
      "prediction": ["predict_promotion_lift"],
      "optimization": ["optimize_promotion_budget"],
      "analysis": ["quick_promotion_analysis"]
    }
  }
}
EOF

echo "✅ Configuration installed to: $CONFIG_FILE"
echo ""
echo "📋 Next steps:"
echo "1. Restart Claude Desktop to load the new configuration"
echo "2. Start a conversation and test with: 'What can this TPM system do?'"
echo "3. Try ML predictions: 'Predict lift for Cheerios with 25% discount for 2 weeks'"
echo ""
echo "🔍 Troubleshooting:"
echo "- If connection fails, check that uv is installed: 'which uv'"
echo "- Verify the project directory path: $PROJECT_DIR"
echo "- Check Claude Desktop logs for detailed error messages"