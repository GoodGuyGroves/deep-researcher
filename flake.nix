{
  description = "Deep research pipeline — researches topics and ingests into OpenViking";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            pkgs.git
          ];

          shellHook = ''
            echo ""
            echo "  Deep Researcher (docker-compose):"
            echo "    docker compose up -d       # start MCP server"
            echo "    docker compose ps          # show status"
            echo "    docker compose down        # stop"
            echo "    docker compose logs -f     # follow logs"
            echo ""
            echo "  CLI (local):"
            echo "    python research.py \"topic\"  # run research directly"
            echo ""
          '';
        };
      }
    );
}
