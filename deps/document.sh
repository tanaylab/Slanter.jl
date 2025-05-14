#!/bin/bash
set -e -o pipefail
julia --color=no deps/document.jl
sed -i 's: on <span class="colophon-date" title="[^"]*">[^<]*</span>::;s:<:\n<:g' docs/v0.1.0/*html
sed -i -E ':a ; $!N ; s/\n<sub>/<sub>/ ; s/\n<\/sub>/<\/sub>/ ; ta ; P ; D' docs/v0.1.0/*html
rm -rf docs/*/*.{cov,jl}
