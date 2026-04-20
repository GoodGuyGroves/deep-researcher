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
            echo "  Usage:  python research.py \"your topic here\""
            echo "  Batch:  python research.py --file topics/my-topics.txt"
            echo ""
          '';
        };
      }
    );
}
