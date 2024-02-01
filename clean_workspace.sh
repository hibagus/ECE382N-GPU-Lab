echo "Workspace Cleaner"
echo "This will return the state of the workspace before you run prepare_workspace.sh"
echo "This action is irreversible and will delete all of your generated data!!!"
read -p "Are you sure (y/n)?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Cleaning-up Workspace..."
    rm -rf nsight-compute/*
    rm -rf application/val
    rm -rf application/train
    rm -rf application/test
    rm -rf application/log
    rm -rf application/checkpoint
    rm -rf application/venv
    cd kernel/build
    make clean
    rm -rf *
    echo "Workspace is now clean."
else
    echo "Clean-up cancelled."
fi
