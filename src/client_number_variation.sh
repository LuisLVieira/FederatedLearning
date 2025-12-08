
for i in {1..2}
do
    echo "Running experiment with $i clients..."
    python main.py --config config/config_"$i"_clients.json
    echo "Experiment with $i clients completed."
done