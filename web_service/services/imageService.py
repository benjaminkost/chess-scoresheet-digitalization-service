import aiofiles
from fastapi import UploadFile

class ImageService:
    def __init__(self, file: UploadFile):
        self.file = file
    async def store_image(self):
        if not (".png" in self.file.filename or ".jpeg" in self.file.filename or ".jpg" in self.file.filename):
            return "This is not a image, the file has to of type: .png, jpeg or jpg"
        else:
            async with aiofiles.open(f"/app/uploads/{self.file.filename}", "wb") as f:
                content = await self.file.read()
                await f.write(content)
            return "File was saved successfully"

    async def create_pgn_file(self, response):
        if response == "File was saved successfully":
            # create PGN File
            pgn_file_str = """
            [Result "1-0"]
            [Date "2023.03.06"]
            [Round "1"]
            [Event "?"]
            [Black "?"]
            [Site "Earth"]
            [White "?"]
            
            1. e4 c6 2. d4 d5 3. exd5 cxd5 4. Nc3 Nc6 5. Bb5 Nf6 6. Nge2 Bg4 7. O-O e6 8. f3 Bf5 9. a3 a6 10. Ba4 b5 
            11. Bb3 Be7 12. Ng3 Bg6 13. f4 Qb6 14. Nce2 Nh5 15. c3 O-O 16. f5 Nxg3 17. Nxg3 exf5 18. Nxf5 Rad8 
            19. Nxe7+ Nxe7 20. Bg5 f6 21. Bd2 Rfe8 22. Qg4 f5 23. Qg5 a5 24. Bc2 h6 25. Qh4 f4 26. Bxg6 Nxg6
            """

            # Write pgn file into the directory
            file_path = f"/app/pgn_files/{self.file.filename}.pgn"
            async with aiofiles.open(file_path, "w") as f:
                await f.write(pgn_file_str)

            return file_path
        else:
            return "No file to process"

