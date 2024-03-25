def convert_xyz_file(input_file_path, output_file_path):
    # Conversion factors
    kcal_to_eV = 0.0433641  # 1 kcal/mol = 0.0433641 eV

    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():  # Start of a new configuration
            new_lines.append(line)  # Number of atoms
            i += 1
            energy = float(lines[i].strip()) * kcal_to_eV  # Convert energy to eV
            header = f'Properties=species:S:1:pos:R:3:forces:R:3 energy={energy} pbc="F F F"'
            new_lines.append(header)
        else:
            # Convert forces if they are present in the line
            if len(line.split()) == 7:  # Check if line has XYZ positions and forces
                elements = line.split()
                el, x, y, z = elements[:4]  # Positions
                fx, fy, fz = [float(f) * kcal_to_eV for f in elements[4:]]  # Convert forces
                new_line = f"{el} {x} {y} {z} {fx} {fy} {fz}"
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        i += 1

    # Write the new data to the output file
    with open(output_file_path, 'w') as file:
        for line in new_lines:
            file.write(line + '\n')

    return "File conversion completed."

convert_xyz_file('train.xyz', 'train_converted.xyz')
