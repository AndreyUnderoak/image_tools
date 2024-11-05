import argparse
import os
import time
import pyodm

if __name__ == "__main__":
    print("First task: stiching images into one\n\n\n")
    # Parsing
    parser = argparse.ArgumentParser(description="Create orthophoto from images in a directory")
    parser.add_argument("directory_path", type=str, help="Path to the directory with images")
    parser.add_argument("--scale_factor", default= 1, type=float, help="Factor by which to scale down images (e.g., 2 for half size)")
    parser.add_argument("--orthophoto_resolution", default= 0.1, type=float, help="Factor by which to scale down images (e.g., 2 for half size)")
    parser.add_argument("--node_adress", default= "localhost", type=str, help="Factor by which to scale down images (e.g., 2 for half size)")
    parser.add_argument("--node_port", default=3000, type=int, help="Factor by which to scale down images (e.g., 2 for half size)")
    args = parser.parse_args()
    # Declare api class

    if args.directory_path[-1] != '/': 
        args.directory_path += '/'
    l = os.listdir(args.directory_path)
    l = [args.directory_path + s for s in l]

    node = pyodm.Node(args.node_adress, args.node_port)

    try:
        # Start a task
        task = node.create_task(l, {'orthophoto-resolution': args.orthophoto_resolution,
                                    'dsm': True, 'skip-report':True,
                                    'fast-orthophoto':True,
                                    'auto-boundary':True,
                                    'skip-3dmodel':True,
                                    'optimize-disk-space': True})
        print(task.info())

        try:
            # This will block until the task is finished
            # or will raise an exception
            print("\nTask INFO:")
            i = ""
            while(task.info().status != pyodm.types.TaskStatus.COMPLETED and task.info().status != pyodm.types.TaskStatus.FAILED):
                print ("\033[A                             \033[A")
                
                i += "."
                if i == ".....":
                    i = ""
                    
                print(str(task.info().status) + " " + str(task.info().progress) + "% " + i)
                time.sleep(0.3)
            # task.wait_for_completion()

            print("Task completed, downloading results...")

            # Retrieve results
            task.download_assets("./odm_media/results")

            print("Assets saved in ./odm_media/results (%s)" % os.listdir("./odm_media/results"))

        except pyodm.exceptions.TaskFailedError as e:
            print("\n".join(task.output()))

    except pyodm.exceptions.NodeConnectionError as e:
        print("Cannot connect: %s" % e)
    except pyodm.exceptions.NodeResponseError as e:
        print("Error: %s" % e)
        
    