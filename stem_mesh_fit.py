from mesh_ops import load_mesh, save_mesh, fit_mesh
def main():
    startingVerts, startingFaces = load_mesh('models/stem_template.obj')
    targetVerts, targetFaces = load_mesh('models/stem_1.obj')
    startingVerts.requires_grad_()
    newVerts, newFaces = fit_mesh(targetVerts, targetFaces, startingVerts, startingFaces, epochs=5000, lr=0.01, logs_path='logs_stem')
    save_mesh(newVerts, newFaces, 'out/stem_fit.obj')
    # python c:\ProgramData\anaconda3\envs\kaolin\Lib\site-packages\kaolin\experimental\dash3d\run.py --logdir=logs-stem

if __name__ == "__main__":
    main()