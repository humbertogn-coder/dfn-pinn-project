# DFN PINN Project

Proyecto de investigacion para implementar y evaluar PINNs para el modelo Doyle-Fuller-Newman completo, con enfoque en:

- referencia reproducible con PyBaMM
- formulacion mixed-variable con corriente interfacial aprendida
- inverse Butler-Volmer como residual de acondicionamiento
- conservacion dura de corriente por proyeccion
- identificabilidad antes de estimar parametros

## Estado actual

Este repositorio empieza con un esqueleto minimo. La primera meta no es entrenar un PINN, sino preparar un flujo reproducible:

1. ambiente local
2. Git y GitHub
3. Codex para desarrollo asistido
4. HPRC Grace para ejecuciones mas pesadas

## Crear ambiente local

En Anaconda Prompt:

```bat
cd "%USERPROFILE%\OneDrive\Documents\ChatGPT\DFN_PINN_Project"
conda create -n dfn-pinn python=3.11 -y
conda activate dfn-pinn
pip install -r requirements.txt
pip install -e .
```

## Verificar instalacion

```bat
python scripts/check_project.py
pytest -q
```

## Flujo con Git

```bat
git init
git add .
git commit -m "Inicializa proyecto DFN PINN"
```

Despues crea un repositorio vacio en GitHub y conecta el remoto:

```bat
git branch -M main
git remote add origin URL_DEL_REPOSITORIO
git push -u origin main
```

## HPRC Grace

Flujo base usando el ambiente existente:

```bash
module purge
module load GCCcore/13.2.0
module load Python/3.11.5
source /scratch/user/humbertogn/ase_env/bin/activate
cd /scratch/user/humbertogn/proyectos/dfn-pinn-project
pip install -r requirements.txt
pip install -e .
python scripts/check_project.py
pytest -q
```

## Primera meta tecnica

Crear un exportador reproducible:

```bash
python scripts/export_pybamm_reference.py --protocol 1C
```

Ese script todavia no existe. Sera el primer modulo real del proyecto.
