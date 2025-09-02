import yaml
import os
import shutil
from jinja2 import Environment, FileSystemLoader
import argparse
import sys
import stat

def generate_case(param_file, output_dir, template_dir='case-template', precice_config_file='../2d/precice-config.xml'):
    """
    Generates a complete, runnable OpenFOAM case from a parameter file.
    """
    print(f"--- Starting OpenFOAM Case Generation ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make paths absolute to be robust
    param_file_abs = os.path.abspath(param_file)
    output_dir_abs = os.path.abspath(output_dir)
    template_dir_abs = os.path.join(script_dir, template_dir)
    precice_config_src_abs = os.path.join(script_dir, precice_config_file)

    print(f"Reading parameters from: {param_file_abs}")

    # Load parameters from YAML file
    try:
        with open(param_file_abs, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Could not read or parse YAML file {param_file_abs}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Using template directory: {template_dir_abs}")
    print(f"Writing case to: {output_dir_abs}")

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(template_dir_abs), trim_blocks=True, lstrip_blocks=True)

    # Walk through the template directory and copy/render files
    for root, dirs, files in os.walk(template_dir_abs):
        relative_dir = os.path.relpath(root, template_dir_abs)
        output_root = os.path.join(output_dir_abs, relative_dir) if relative_dir != '.' else output_dir_abs
        os.makedirs(output_root, exist_ok=True)

        for file in files:
            template_path = os.path.join(root, file)
            output_path = os.path.join(output_root, file.replace('.template', ''))

            if file.endswith('.template'):
                template_rel_path = os.path.relpath(template_path, template_dir_abs)
                template = env.get_template(template_rel_path)
                rendered_content = template.render(params)
                with open(output_path, 'w') as f:
                    f.write(rendered_content)
            else:
                shutil.copy2(template_path, output_path)

    # Copy the preCICE configuration file into the case's system directory
    precice_config_dest = os.path.join(output_dir_abs, 'precice-config.xml')
    print(f"Copying preCICE config from {precice_config_src_abs} to {precice_config_dest}")
    shutil.copy2(precice_config_src_abs, precice_config_dest)
    
    print(f"--- Case generation complete for: {output_dir_abs} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an OpenFOAM case from a parameter file.")
    parser.add_argument("param_file", type=str, help="Path to the YAML parameter file.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the case will be generated.")
    
    args = parser.parse_args()
    generate_case(args.param_file, args.output_dir)
