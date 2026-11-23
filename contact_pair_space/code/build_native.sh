#!/bin/sh
# Builds the exact sweep (native/vertical_lift.cpp) as a shared library in contact_pair/_native/. Needs a C++17 compiler.
# macOS: libvertical_lift.dylib; Linux and others: libvertical_lift.so.
set -e
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
case "$(uname -s)" in
  Darwin) OUT=contact_pair/_native/libvertical_lift.dylib; "$CXX" -O3 -std=c++17 -dynamiclib -o "$OUT" native/vertical_lift.cpp ;;
  *)      OUT=contact_pair/_native/libvertical_lift.so;    "$CXX" -O3 -std=c++17 -shared -fPIC -pthread -o "$OUT" native/vertical_lift.cpp ;;
esac
echo "built $OUT"
