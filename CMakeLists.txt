project(forthenon LANGUAGES CXX)

set(PARTHENON_IMPORT_KOKKOS ON)
add_subdirectory(extern/parthenon)

add_executable(
   advection_forthenon
   main.cpp
   forthenon.cpp
   advection.cpp
   )

target_link_libraries(advection_forthenon PRIVATE Parthenon::parthenon)

